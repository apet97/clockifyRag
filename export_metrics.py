#!/usr/bin/env python3
"""
Export metrics from the Clockify RAG system.

Priority #13: Export KPI metrics (ROI 5/10)

Usage:
    # Export as JSON
    python3 export_metrics.py --format json

    # Export as Prometheus
    python3 export_metrics.py --format prometheus

    # Export as CSV
    python3 export_metrics.py --format csv

    # Export to file
    python3 export_metrics.py --format json --output metrics.json

    # Show summary only
    python3 export_metrics.py --summary
"""

import argparse
import sys
import json

from clockify_rag.metrics import get_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Export metrics from Clockify RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--format",
        choices=["json", "prometheus", "csv"],
        default="json",
        help="Export format (default: json)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics only"
    )
    parser.add_argument(
        "--no-histograms",
        action="store_true",
        help="Exclude raw histogram data from JSON export (reduces size)"
    )

    args = parser.parse_args()

    metrics = get_metrics()

    # Generate output
    if args.summary:
        summary = metrics.get_summary()
        output = json.dumps(summary, indent=2)
    elif args.format == "json":
        output = metrics.export_json(include_histograms=not args.no_histograms)
    elif args.format == "prometheus":
        output = metrics.export_prometheus()
    elif args.format == "csv":
        output = metrics.export_csv()
    else:
        print(f"Error: Unknown format '{args.format}'", file=sys.stderr)
        return 1

    # Write output
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Metrics exported to: {args.output}", file=sys.stderr)
        except IOError as e:
            print(f"Error writing to {args.output}: {e}", file=sys.stderr)
            return 1
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
