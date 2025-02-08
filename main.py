"""Module for distributed OCR using PySpark."""

import sys
from io import BytesIO
import pytesseract # pylint: disable=import-error
from pyspark.sql import SparkSession # pylint: disable=import-error
from PIL import Image # pylint: disable=import-error


def split_image(image_path, num_splits):
    """Splits an image into a specified number of horizontal parts.

    This function takes an input image and divides it into equal horizontal sections,
    with the last section potentially being slightly larger to account for any remainder.

    Args:
        image_path (str): The file path to the input image.
        num_splits (int): The number of horizontal sections to split the image into.

    Returns:
        list: A list of PIL.Image objects representing the horizontal splits of the original image.
              Each split maintains the original width but has a height of approximately
              (original_height / num_splits).
    """
    img = Image.open(image_path)
    width, height = img.size
    split_height = height // num_splits
    splits = []
    for i in range(num_splits):
        top = i * split_height
        bottom = (i + 1) * split_height if i < num_splits - 1 else height
        region = img.crop((0, top, width, bottom))
        splits.append(region)
    return splits


def image_to_byte_array(image):
    """
    Convert a PIL Image object to a byte array.

    This function takes a PIL Image instance and converts it to a byte array in PNG format.
    The image is first saved to a BytesIO buffer and then converted to bytes.

    Args:
        image (PIL.Image.Image): The PIL Image object to be converted

    Returns:
        bytes: The image data as a byte array in PNG format
    """
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def ocr_process(byte_data):
    """
    Performs Optical Character Recognition (OCR) on image data.

    Args:
        byte_data (bytes): Raw image data in bytes format.

    Returns:
        str: Extracted text from the image.

    Raises:
        PIL.UnidentifiedImageError: If the byte data cannot be opened as an image.
        
    Note:
        This function uses pytesseract for OCR processing and requires the tesseract-ocr
        engine to be installed on the system.
    """
    image = Image.open(BytesIO(byte_data))
    text = pytesseract.image_to_string(image)
    return text


if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: spark_ocr.py <image_path> <output> <num_splits>")
        sys.exit(1)

    image_path_arg = sys.argv[1]
    output_path_arg = sys.argv[2]
    num_splits_arg = int(sys.argv[3]) if len(sys.argv) == 4 else 4

    spark = (
        SparkSession.builder.appName("DistributedOCR")
        .config("spark.executor.memory", "500m")
        .config("spark.executor.cores", "1")
        .getOrCreate()
    )

    try:
        image_splits = split_image(image_path_arg, num_splits_arg)
        split_bytes = [image_to_byte_array(split) for split in image_splits]

        rdd = spark.sparkContext.parallelize(split_bytes, numSlices=num_splits_arg)
        ocr_results = rdd.map(ocr_process).collect()

        FULL_TEXT = "\n".join(ocr_results)
        print(FULL_TEXT)

        with open(output_path_arg, "w", encoding='utf-8') as f:
            f.write(FULL_TEXT)

    finally:
        spark.stop()
