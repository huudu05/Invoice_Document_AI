from pathlib import Path
from dataclasses import dataclass, field


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

MAX_FILE_SIZE_MB = 20

MIN_IMAGE_WIDTH = 300
MIN_IMAGE_HEIGHT = 300



@dataclass
class ValidationResult:

    is_valid: bool
    message: str
    file_type: str | None = None
    file_size: int = 0
    warnings: list[str] = field(default_factory=list)

class FileValidator:
    """
    Validate the invoice before processing
    """

    def __init__(self, max_file_size_mb: int = MAX_FILE_SIZE_MB, ):
        self.max_file_size_mb = max_file_size_mb

    def validate(self, file_path: str | Path) -> ValidationResult:
        file_path = Path(file_path)

        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                message="File does not exist."
            )

        suffix = file_path.suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                message=f"Unsupported file type: {suffix}"
            )

        file_size = file_path.stat().st_size

        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return ValidationResult(
                is_valid=False,
                message=f"file exceeds {self.max_file_size_mb} MB."
            )

        return ValidationResult(
            is_valid=True,
            message="Validation successful.",
            file_type=suffix,
            file_size=file_size
        )
    
        