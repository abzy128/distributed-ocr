"""Module for distributed OCR using PySpark."""

import sys
import os
from io import BytesIO
import pytesseract # pylint: disable=import-error
from pyspark.sql import SparkSession # pylint: disable=import-error
from PIL import Image, UnidentifiedImageError # pylint: disable=import-error


def log(msg, level="info"):
    """Log a message to the console."""
    level_color = "\033[0m"
    if level == "info":
        level_color = "\033[96m"
    elif level == "warning":
        level_color = "\033[93m"
    elif level == "error":
        level_color = "\033[91m"

    print(level_color + msg + "\033[0m")


def ocr_process(img_bytes):
    """Performs OCR on image bytes."""
    try:
        image = Image.open(BytesIO(img_bytes))
        ocr_output = pytesseract.image_to_string(image)
        return ocr_output
    except (UnidentifiedImageError, pytesseract.TesseractError) as e:
        log(f"Error during OCR: {e}", level="error")
        return ""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        log("Usage: spark_ocr.py <input_directory>", level="error")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = os.path.join(input_dir, "output")

    spark = (
        SparkSession.builder.appName("DistributedOCR")
        .config("spark.executor.memory", "500m")
        .config("spark.executor.cores", "1")
        .getOrCreate()
    )

    try:
        image_files = []
        for f in os.listdir(input_dir):
            full_path = os.path.join(input_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".tif")
            ):
                try:
                    with open(full_path, "rb") as img_file:
                        image_bytes = img_file.read()
                        image_files.append((f, image_bytes))
                except (IOError, PermissionError) as e:
                    log(f"Error reading image {f}: {e}", level="error")

        def process_image(image_data):
            """Process an image using OCR to extract text.

            Args:
                image_data (tuple): A tuple containing image filename (str) and image data (bytes).

            Returns:
                tuple: A tuple containing:
                    - str or None: The image filename if successful, None if failed
                    - str: The extracted text if successful, empty string if failed

            Raises:
                pytesseract.TesseractError: If OCR processing fails, error is caught and logged
            """
            image_filename, image_data_bytes = image_data
            try:
                extracted_text = ocr_process(image_data_bytes)
                return image_filename, extracted_text
            except pytesseract.TesseractError as e:
                log(f"Error processing image {image_filename}: {e}", level="error")
                return None, ""

        rdd = spark.sparkContext.parallelize(image_files)
        ocr_results = rdd.map(process_image).collect()

        for filename, text in ocr_results:
            if filename:  # Check if filename is valid (not None due to error)
                output_file = os.path.join(
                    output_dir, f"{os.path.splitext(filename)[0]}.txt"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text)
                log(f"OCR results for {filename} saved to {output_file}")

    except IOError as e:
        log(f"File operation error: {e}")
    except RuntimeError as e:
        log(f"Spark operation error: {e}")

    finally:
        log("Distirbuted OCR complete.")
        log(f"You can find the OCR results in: {output_dir}")
        log("Spark context is stil available at http://localhost:4040")
        log("Press Ctrl+C to exit.")
        input()
        spark.stop()
