import csv
import json
from pathlib import Path

from .models import OCRResult


class OCRExporter:

    def to_dict(self, result: OCRResult):
        pages = []

        for page in result.pages:
            words = []

            for word in page.words:
                words.append({
                    "id": word.id,
                    "text": word.text,
                    "confidence": word.confidence,
                    "bbox": word.bbox
                })

            pages.append({
                "page_number": page.page_number,
                "words": words
            })

        return {"pages": pages}

    def to_json(self, result: OCRResult):

        return json.dumps(
            self.to_dict(result),
            indent=4,
            ensure_ascii=False
        )

    def to_text(self, result: OCRResult):
        lines = []

        for page in result.pages:
            lines.append(
                f"PAGE {page.page_number}"
            )

            for word in page.words:
                lines.append(word.text)

            lines.append("")

        return "\n".join(lines)

    def to_csv_rows(self, result: OCRResult):
        rows = []

        for page in result.pages:
            for word in page.words:
                rows.append(
                    [
                        page.page_number,
                        word.id,
                        word.text,
                        word.confidence,
                        word.bbox
                    ]
                )

        return rows

    def export_json(
        self,
        result,
        output_path
    ):

        Path(output_path).write_text(
            self.to_json(result),
            encoding="utf-8"
        )

    def export_txt(
        self,
        result,
        output_path
    ):

        Path(output_path).write_text(
            self.to_text(result),
            encoding="utf-8"
        )

    def export_csv(
        self,
        result,
        output_path
    ):

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "page",
                    "id",
                    "text",
                    "confidence",
                    "bbox"
                ]
            )

            writer.writerows(
                self.to_csv_rows(result)
            )