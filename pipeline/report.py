from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd


@dataclass
class CleaningReport:
    original_shape: tuple = (0, 0)
    final_shape: tuple = (0, 0)
    dropped_columns: List[str] = field(default_factory=list)
    imputed_columns: Dict[str, str] = field(default_factory=dict)
    encoded_columns: Dict[str, str] = field(default_factory=dict)
    outlier_actions: Dict[str, str] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    task_detected: str = ""
    step_timings: Dict[str, float] = field(default_factory=dict)  # Step 8
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Original shape  : {self.original_shape}",
            f"Final shape     : {self.final_shape}",
            f"Task detected   : {self.task_detected}",
            f"Dropped columns : {self.dropped_columns or 'none'}",
            f"Imputed columns : {list(self.imputed_columns.keys()) or 'none'}",
            f"Encoded columns : {self.encoded_columns or 'none'}",
            f"Outlier actions : {list(self.outlier_actions.keys()) or 'none'}",
            f"Selected feats  : {self.selected_features or 'none'}",
        ]
        if self.step_timings:
            lines.append("Step timings    :")
            for step, t in self.step_timings.items():
                lines.append(f"  {step:<20} {t:.3f}s")
        if self.warnings:
            lines += [f"--> {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_html(self) -> str:
        """Export report as a self-contained HTML string for download."""
        rows = {
            "Original shape":  str(self.original_shape),
            "Final shape":     str(self.final_shape),
            "Task detected":   self.task_detected,
            "Dropped columns": ", ".join(self.dropped_columns) or "none",
            "Imputed columns": ", ".join(self.imputed_columns.keys()) or "none",
            "Encoded columns": ", ".join(
                f"{k} ({v})" for k, v in self.encoded_columns.items()
            ) or "none",
            "Outlier actions": ", ".join(
                f"{k} ({v})" for k, v in self.outlier_actions.items()
            ) or "none",
            "Selected features": ", ".join(self.selected_features) or "none",
        }
        table_rows = "\n".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows.items()
        )
        timing_rows = ""
        if self.step_timings:
            timing_rows = "<h3>Step timings</h3><table>" + "\n".join(
                f"<tr><td>{s}</td><td>{t:.3f}s</td></tr>"
                for s, t in self.step_timings.items()
            ) + "</table>"
        warnings_html = ""
        if self.warnings:
            items = "\n".join(f"<li>{w}</li>" for w in self.warnings)
            warnings_html = f"<h3>Warnings</h3><ul>{items}</ul>"

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cleaning Report</title>
<style>
  body  {{ font-family: sans-serif; max-width: 720px; margin: 40px auto; color: #222; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  td    {{ padding: 8px 12px; border: 1px solid #ddd; vertical-align: top; }}
  tr:nth-child(even) {{ background: #f7f7f7; }}
  h1    {{ font-size: 22px; }} h3 {{ font-size: 16px; margin-top: 28px; }}
</style>
</head>
<body>
<h1>Cleaning Report</h1>
<table>{table_rows}</table>
{timing_rows}
{warnings_html}
</body>
</html>"""