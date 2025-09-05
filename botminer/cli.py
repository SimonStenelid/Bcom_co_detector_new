"""Command-line interface for botminer."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
import logging
from datetime import datetime

from .config import Config
from .ingest import ingest_csv
from .ingest_batch import ingest_all_data
from .features import build_features
from .scoring import score_cohorts
from .report import generate_reports

app = typer.Typer(
    name="botminer",
    help="Batch pattern miner for detecting bot traffic in flight search logs",
    no_args_is_help=True,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.command()
def ingest(
    csv_path: str = typer.Argument(..., help="Path to input CSV file"),
    date: str = typer.Argument(..., help="Date in YYYY-MM-DD format"),
    config_path: str = typer.Option("config.yaml", help="Path to config file"),
    ip_csv_path: str = typer.Option(None, "--ip-csv", help="Path to IP address CSV file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Ingest CSV data and store as Parquet."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        config = Config.from_file(config_path)
        logger.info(f"Ingesting CSV: {csv_path} for date: {date}")
        
        result = ingest_csv(csv_path, date, config, ip_csv_path)
        logger.info(f"Successfully ingested {result['rows_processed']} rows")
        logger.info(f"Output saved to: {result['output_path']}")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise typer.Exit(1)


@app.command()
def features(
    date: str = typer.Argument(..., help="Date in YYYY-MM-DD format"),
    config_path: str = typer.Option("config.yaml", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Build cohort-level features from raw data."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        config = Config.from_file(config_path)
        logger.info(f"Building features for date: {date}")
        
        result = build_features(date, config)
        logger.info(f"Successfully built features for {result['cohorts_processed']} cohorts")
        logger.info(f"Output saved to: {result['output_path']}")
        
    except Exception as e:
        logger.error(f"Error during feature building: {e}")
        raise typer.Exit(1)


@app.command()
def score(
    date: str = typer.Argument(..., help="Date in YYYY-MM-DD format"),
    config_path: str = typer.Option("config.yaml", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Score cohorts for botness using rule-based and anomaly detection."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        config = Config.from_file(config_path)
        logger.info(f"Scoring cohorts for date: {date}")
        
        result = score_cohorts(date, config)
        logger.info(f"Successfully scored {result['cohorts_scored']} cohorts")
        logger.info(f"High risk: {result['high_risk']}, Medium risk: {result['medium_risk']}, Low risk: {result['low_risk']}")
        logger.info(f"Output saved to: {result['output_path']}")
        
    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        raise typer.Exit(1)


@app.command()
def report(
    date: str = typer.Argument(..., help="Date in YYYY-MM-DD format"),
    config_path: str = typer.Option("config.yaml", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Generate reports (CSV, JSON, Markdown) for scored cohorts."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        config = Config.from_file(config_path)
        logger.info(f"Generating reports for date: {date}")
        
        result = generate_reports(date, config)
        logger.info(f"Successfully generated reports")
        logger.info(f"CSV: {result['csv_path']}")
        logger.info(f"JSON: {result['json_path']}")
        logger.info(f"Markdown: {result['md_path']}")
        
    except Exception as e:
        logger.error(f"Error during report generation: {e}")
        raise typer.Exit(1)


@app.command()
def pipeline(
    date: str = typer.Argument(..., help="Date in YYYY-MM-DD format"),
    csv_path: str = typer.Option(None, help="Path to input CSV file (optional for batch mode)"),
    config_path: str = typer.Option("config.yaml", help="Path to config file"),
    ip_csv_path: str = typer.Option(None, "--ip-csv", help="Path to IP address CSV file"),
    batch_mode: bool = typer.Option(False, "--batch", help="Process all files in data/raw folder"),
    data_dir: str = typer.Option("data/raw", help="Data directory for batch mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run the complete pipeline: ingest -> features -> score -> report."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        config = Config.from_file(config_path)
        logger.info(f"Running complete pipeline for date: {date}")
        
        # Step 1: Ingest
        logger.info("🔄 Step 1/4: Data Ingestion...")
        if batch_mode:
            logger.info("📁 Using batch mode - processing all files in data directory")
            ingest_result = ingest_all_data(data_dir, date, config)
            logger.info(f"✅ Ingestion Complete: {ingest_result['rows_processed']:,} rows processed")
            logger.info(f"   📊 Data sources: {len(ingest_result['booking_files'])} booking + {len(ingest_result['ip_files'])} IP files")
            if 'data_reduction_ratio' in ingest_result:
                logger.info(f"   🎯 Data quality: {ingest_result['data_reduction_ratio']*100:.1f}% of records had matching IPs")
        else:
            if csv_path is None:
                raise ValueError("CSV path is required when not using batch mode")
            ingest_result = ingest_csv(csv_path, date, config, ip_csv_path)
            logger.info(f"✅ Ingested {ingest_result['rows_processed']:,} rows")
        
        # Step 2: Features
        logger.info("🔄 Step 2/4: Feature Engineering...")
        features_result = build_features(date, config)
        logger.info(f"✅ Built features for {features_result['cohorts_processed']:,} cohorts")
        
        # Step 3: Scoring
        logger.info("🔄 Step 3/4: Bot Detection Scoring...")
        score_result = score_cohorts(date, config)
        logger.info(f"✅ Scored {score_result['cohorts_scored']:,} cohorts")
        logger.info(f"   🚨 High risk: {score_result['high_risk']}")
        logger.info(f"   ⚠️  Medium risk: {score_result['medium_risk']}")
        logger.info(f"   ✅ Low risk: {score_result['low_risk']}")
        
        # Step 4: Reports
        logger.info("🔄 Step 4/4: Generating reports...")
        report_result = generate_reports(date, config)
        logger.info("✅ Generated all reports")
        
        logger.info("🎉 Pipeline completed successfully!")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"📋 Reports available in: reports/{date}_{timestamp}")
        
        # Final summary
        if batch_mode and 'unique_cohorts' in ingest_result:
            logger.info(f"\n📈 Final Analysis Summary:")
            logger.info(f"   📊 Processed {ingest_result['rows_processed']:,} events")
            logger.info(f"   🎯 Identified {ingest_result['unique_cohorts']:,} unique cohorts")
            logger.info(f"   🌐 From {ingest_result['unique_ips']:,} unique IP addresses")
            logger.info(f"   🔧 Using {ingest_result['unique_user_agents']:,} different user agents")
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
