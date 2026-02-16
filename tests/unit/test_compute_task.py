"""Unit tests for fractal._compute_task."""

from unittest.mock import MagicMock, patch

import pytest

from ome_zarr_converters_tools.fractal._compute_task import (
    _build_image_list_update,
    _format_attribute_value,
    generic_compute_task,
)
from ome_zarr_converters_tools.fractal._models import ConvertParallelInitArgs
from ome_zarr_converters_tools.models import (
    ConverterOptions,
    ImageInPlate,
    SingleImage,
)


class TestFormatAttributeValue:
    def test_single_value_returns_scalar(self) -> None:
        assert _format_attribute_value(["DMSO"]) == "DMSO"

    def test_single_numeric_value(self) -> None:
        assert _format_attribute_value([42]) == 42

    def test_multiple_values_joined(self) -> None:
        result = _format_attribute_value(["A", "B", "C"])
        assert result == "A & B & C"

    def test_multiple_numeric_values_joined(self) -> None:
        result = _format_attribute_value([1, 2])
        assert result == "1 & 2"

    def test_single_none_value(self) -> None:
        assert _format_attribute_value([None]) is None

    def test_single_bool_value(self) -> None:
        assert _format_attribute_value([True]) is True


class TestBuildImageListUpdate:
    def _make_ome_zarr_mock(
        self, *, is_3d: bool = False, is_time_series: bool = False
    ) -> MagicMock:
        mock = MagicMock()
        mock.is_3d = is_3d
        mock.is_time_series = is_time_series
        return mock

    def test_basic_structure(self) -> None:
        ome_zarr = self._make_ome_zarr_mock()
        collection = SingleImage(image_path="test")
        result = _build_image_list_update(
            zarr_url="/tmp/test.zarr",
            ome_zarr=ome_zarr,
            collection=collection,
            attributes={},
        )
        assert "image_list_updates" in result
        assert len(result["image_list_updates"]) == 1
        update = result["image_list_updates"][0]
        assert update["zarr_url"] == "/tmp/test.zarr"

    def test_3d_type(self) -> None:
        ome_zarr = self._make_ome_zarr_mock(is_3d=True)
        result = _build_image_list_update(
            zarr_url="/tmp/test.zarr",
            ome_zarr=ome_zarr,
            collection=SingleImage(image_path="test"),
            attributes={},
        )
        assert result["image_list_updates"][0]["types"]["is_3D"] is True

    def test_time_series_type(self) -> None:
        ome_zarr = self._make_ome_zarr_mock(is_time_series=True)
        result = _build_image_list_update(
            zarr_url="/tmp/test.zarr",
            ome_zarr=ome_zarr,
            collection=SingleImage(image_path="test"),
            attributes={},
        )
        assert result["image_list_updates"][0]["types"]["is_time_series"] is True

    def test_no_time_series_key_absent(self) -> None:
        ome_zarr = self._make_ome_zarr_mock(is_time_series=False)
        result = _build_image_list_update(
            zarr_url="/tmp/test.zarr",
            ome_zarr=ome_zarr,
            collection=SingleImage(image_path="test"),
            attributes={},
        )
        assert "is_time_series" not in result["image_list_updates"][0]["types"]

    def test_attributes_formatted(self) -> None:
        ome_zarr = self._make_ome_zarr_mock()
        result = _build_image_list_update(
            zarr_url="/tmp/test.zarr",
            ome_zarr=ome_zarr,
            collection=SingleImage(image_path="test"),
            attributes={"drug": ["DMSO"], "dose": [1.0, 2.0]},
        )
        attrs = result["image_list_updates"][0]["attributes"]
        assert attrs["drug"] == "DMSO"
        assert attrs["dose"] == "1.0 & 2.0"

    def test_image_in_plate_attributes(self) -> None:
        ome_zarr = self._make_ome_zarr_mock()
        collection = ImageInPlate(
            plate_name="MyPlate", row="A", column=1, acquisition=0
        )
        result = _build_image_list_update(
            zarr_url="/tmp/test.zarr",
            ome_zarr=ome_zarr,
            collection=collection,
            attributes={},
        )
        attrs = result["image_list_updates"][0]["attributes"]
        assert attrs["plate"] == collection.plate_path()
        assert attrs["well"] == collection.well
        assert attrs["acquisition"] == collection.acquisition

    def test_single_image_no_plate_attributes(self) -> None:
        ome_zarr = self._make_ome_zarr_mock()
        result = _build_image_list_update(
            zarr_url="/tmp/test.zarr",
            ome_zarr=ome_zarr,
            collection=SingleImage(image_path="test"),
            attributes={},
        )
        attrs = result["image_list_updates"][0]["attributes"]
        assert "plate" not in attrs
        assert "well" not in attrs


class TestGenericComputeTask:
    @patch(
        "ome_zarr_converters_tools.fractal._compute_task.tiled_image_creation_pipeline"
    )
    @patch(
        "ome_zarr_converters_tools.fractal._compute_task.build_default_registration_pipeline"
    )
    @patch("ome_zarr_converters_tools.fractal._compute_task.remove_json")
    @patch("ome_zarr_converters_tools.fractal._compute_task.tiled_image_from_json")
    def test_successful_compute(
        self,
        mock_from_json: MagicMock,
        mock_remove: MagicMock,
        mock_build_reg: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        # Setup mocks
        mock_tiled_image = MagicMock()
        mock_tiled_image.collection = SingleImage(image_path="test")
        mock_tiled_image.attributes = {}
        mock_from_json.return_value = mock_tiled_image

        mock_ome_zarr = MagicMock()
        mock_ome_zarr.is_3d = False
        mock_ome_zarr.is_time_series = False
        mock_pipeline.return_value = mock_ome_zarr

        mock_build_reg.return_value = []

        init_args = ConvertParallelInitArgs(
            tiled_image_json_dump_url="/tmp/test.json",
            converter_options=ConverterOptions(),
        )

        result = generic_compute_task(
            zarr_url="/tmp/test.zarr",
            init_args=init_args,
            collection_type=SingleImage,
            image_loader_type=MagicMock,
        )

        assert "image_list_updates" in result
        mock_from_json.assert_called_once()
        mock_remove.assert_called_once_with("/tmp/test.json")
        mock_pipeline.assert_called_once()

    @patch("ome_zarr_converters_tools.fractal._compute_task.tiled_image_from_json")
    def test_retries_on_file_not_found(
        self,
        mock_from_json: MagicMock,
    ) -> None:
        mock_from_json.side_effect = FileNotFoundError("not found")

        init_args = ConvertParallelInitArgs(
            tiled_image_json_dump_url="/tmp/missing.json",
            converter_options=ConverterOptions(),
        )

        with (
            patch("ome_zarr_converters_tools.fractal._compute_task.time.sleep"),
            pytest.raises(FileNotFoundError, match="after 3 retries"),
        ):
            generic_compute_task(
                zarr_url="/tmp/test.zarr",
                init_args=init_args,
                collection_type=SingleImage,
                image_loader_type=MagicMock,
            )

        assert mock_from_json.call_count == 3

    @patch(
        "ome_zarr_converters_tools.fractal._compute_task.tiled_image_creation_pipeline"
    )
    @patch(
        "ome_zarr_converters_tools.fractal._compute_task.build_default_registration_pipeline"
    )
    @patch("ome_zarr_converters_tools.fractal._compute_task.remove_json")
    @patch("ome_zarr_converters_tools.fractal._compute_task.tiled_image_from_json")
    def test_retry_succeeds_on_second_attempt(
        self,
        mock_from_json: MagicMock,
        mock_remove: MagicMock,
        mock_build_reg: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        mock_tiled_image = MagicMock()
        mock_tiled_image.collection = SingleImage(image_path="test")
        mock_tiled_image.attributes = {}

        mock_from_json.side_effect = [
            FileNotFoundError("not found"),
            mock_tiled_image,
        ]

        mock_ome_zarr = MagicMock()
        mock_ome_zarr.is_3d = False
        mock_ome_zarr.is_time_series = False
        mock_pipeline.return_value = mock_ome_zarr
        mock_build_reg.return_value = []

        init_args = ConvertParallelInitArgs(
            tiled_image_json_dump_url="/tmp/test.json",
            converter_options=ConverterOptions(),
        )

        with patch("ome_zarr_converters_tools.fractal._compute_task.time.sleep"):
            result = generic_compute_task(
                zarr_url="/tmp/test.zarr",
                init_args=init_args,
                collection_type=SingleImage,
                image_loader_type=MagicMock,
            )

        assert "image_list_updates" in result
        assert mock_from_json.call_count == 2
