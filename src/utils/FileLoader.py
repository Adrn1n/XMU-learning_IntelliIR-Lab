import sys
import os
from charset_normalizer import detect

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Utils.Logger import setup_logger


class FileLoader:
    __logger = setup_logger(__name__)

    @classmethod
    def load_file_content(cls, file_path, encoding=None):
        if not os.path.exists(file_path):
            cls.__logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if encoding:
                cls.__logger.debug(
                    f"Loading {file_path} with specified encoding {encoding}"
                )

                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()

            with open(file_path, "rb") as f:
                raw_data = f.read()

            detected_encoding = detect(raw_data)["encoding"]

            if not detected_encoding:
                cls.__logger.error(f"Failed to detect encoding for {file_path}")
                raise UnicodeDecodeError(
                    "encoding_detection",
                    raw_data,
                    0,
                    len(raw_data),
                    f"Failed to detect encoding for {file_path}",
                )

            file_content = raw_data.decode(detected_encoding)

            cls.__logger.debug(
                f"Successfully loaded {file_path} with detected encoding {detected_encoding}"
            )
            return file_content
        except (IOError, OSError) as e:
            cls.__logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
        except UnicodeDecodeError as e:
            cls.__logger.error(f"Encoding error for {file_path}: {str(e)}")
            raise

    @classmethod
    def load_directory_all_files(
        cls, dir_path, extensions=None, extensions_exclude=None
    ):
        if not os.path.isdir(dir_path):
            cls.__logger.error(f"Directory not found: {dir_path}")
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        abs_dir = os.path.abspath(dir_path)
        results = []

        try:
            for root, _, files in os.walk(abs_dir):

                def numeric_sort_key(filename):
                    name_without_ext = os.path.splitext(filename)[0]

                    try:
                        return int(name_without_ext)
                    except ValueError:
                        return float("inf"), filename

                files.sort(key=numeric_sort_key)

                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    include_file = True

                    if extensions and ext not in extensions:
                        include_file = False
                    elif extensions_exclude and ext in extensions_exclude:
                        include_file = False

                    if include_file:
                        results.append(os.path.join(root, file))

            cls.__logger.info(f"Found {len(results)} files in {abs_dir}")
            return results

        except (OSError, PermissionError) as e:
            cls.__logger.error(f"Error accessing directory {abs_dir}: {str(e)}")
            raise


if __name__ == "__main__":
    # Test FileLoader functionality
    loader = FileLoader()

    # Test loading a directory
    testDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test")
    testExtensions = [".txt", ""]
    testExtensionsExclude = [".md"]

    try:
        paths = loader.load_directory_all_files(
            testDir, testExtensions, testExtensionsExclude
        )
        print(f"Found {len(paths)} files")

        for path in paths[:5]:  # Test first 5 files
            try:
                content = loader.load_file_content(path, None)
                print(f"Loaded {path} with {len(content)} characters.")
                print(content[:100])  # Print first 100 characters
            except Exception as ex:
                print(f"Failed to load {path}: {str(ex)}")
    except Exception as excpt:
        print(f"Directory loading failed: {str(excpt)}")
