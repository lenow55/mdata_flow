import pathlib


class FileNameValidator:
    @staticmethod
    def validate_with_os(filename: str):
        ## Check invalid characters
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        return not any(char in filename for char in invalid_chars)

    @staticmethod
    def validate_with_pathlib(filepath: str):
        try:
            path = pathlib.Path(filepath)
            _ = path.resolve()
            return True
        except Exception:
            return False

    @staticmethod
    def sanitize(filename: str):
        ## Remove or replace invalid characters
        return "".join(
            char if char.isalnum() or char in ["_", "-", "."] else "-"
            for char in filename
        )

    @staticmethod
    def is_valid(filename: str | None, max_length: int = 255):
        ## Comprehensive validation method
        if not filename:
            return False

        if len(filename) > max_length:
            return False

        ## Forbidden names and patterns
        forbidden_names = ["CON", "PRN", "AUX", "NUL"]
        if filename.upper() in forbidden_names:
            return False

        if not FileNameValidator.validate_with_pathlib(filename):
            return False

        if not FileNameValidator.validate_with_os(filename):
            return False

        return True
