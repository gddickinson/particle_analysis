# src/particle_analysis/__main__.py

import sys
import os
import logging
import argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from .gui.main_window import MainWindow

def setup_logging(log_level: str = 'INFO') -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('particle_analysis.log')
        ]
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Particle Analysis Application')
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='Input file to open on startup'
    )
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info('Starting Particle Analysis Application')
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create main window
    window = MainWindow()
    
    # Load configuration if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            logger.info(f'Loading configuration from {config_path}')
            # TODO: Implement configuration loading
    
    # Load input file if provided
    if args.input_file:
        input_path = Path(args.input_file)
        if input_path.exists():
            logger.info(f'Loading input file {input_path}')
            window.open_file(str(input_path))
    
    # Show window and run application
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()