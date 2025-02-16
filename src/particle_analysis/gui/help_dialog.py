# src/particle_analysis/gui/help_dialog.py

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QTextBrowser,
    QPushButton
)
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
import os
from pathlib import Path

class HelpDialog(QDialog):
    """Dialog for displaying help and documentation"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Particle Analysis Help")
        self.setMinimumSize(800, 600)  # Set a reasonable minimum size

        # Get path to docs directory
        package_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        self.docs_dir = package_dir / 'docs'

        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Create tab widget for different help sections
        tabs = QTabWidget()

        # Add all tabs
        tabs.addTab(self.create_quick_start_tab(), "Quick Start")
        tabs.addTab(self.create_parameters_tab(), "Parameters")
        tabs.addTab(self.create_troubleshooting_tab(), "Troubleshooting")
        tabs.addTab(self.create_documentation_tab(), "Documentation")

        layout.addWidget(tabs)

        # Add buttons
        button_box = QHBoxLayout()

        # Report Issue button
        report_btn = QPushButton("Report Issue")
        report_btn.clicked.connect(self.report_issue)
        button_box.addWidget(report_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_box.addWidget(close_btn)

        layout.addLayout(button_box)

    def create_quick_start_tab(self) -> QWidget:
        """Create the Quick Start tab"""
        quick_start = QWidget()
        quick_layout = QVBoxLayout(quick_start)
        quick_text = QTextBrowser()
        quick_text.setOpenExternalLinks(True)
        quick_text.setHtml("""
            <h2>Quick Start Guide</h2>

            <p><b>Basic Workflow:</b></p>
            <ol>
                <li>Open a TIFF stack using File → Open or Ctrl+O</li>
                <li>Choose analysis method:
                    <ul>
                        <li><b>Full Image Analysis:</b> Use "Analyze Full Image" button</li>
                        <li><b>ROI Analysis:</b>
                            <ol>
                                <li>Draw ROI box in the image</li>
                                <li>Click "Detect Particles"</li>
                                <li>Click "Track Particles"</li>
                            </ol>
                        </li>
                    </ul>
                </li>
                <li>View results in:
                    <ul>
                        <li>Visualization tab - Track display</li>
                        <li>Analysis tab - Feature histograms</li>
                        <li>Results Tables tab - Detailed data</li>
                    </ul>
                </li>
                <li>Export results using File → Save Results</li>
            </ol>

            <p><b>Key Features:</b></p>
            <ul>
                <li><b>Particle Overlay:</b> Toggle to verify detection accuracy</li>
                <li><b>Track Visualization:</b> Color-code tracks by various properties</li>
                <li><b>Feature Analysis:</b> Histograms and statistics</li>
                <li><b>Results Tables:</b> Filter, sort, and export data</li>
            </ul>

            <p><b>Tips:</b></p>
            <ul>
                <li>Start with a small ROI to test parameters</li>
                <li>Use particle overlay to verify detection quality</li>
                <li>Adjust parameters based on tracking results</li>
                <li>Save settings when you find good parameters</li>
            </ul>
        """)
        quick_layout.addWidget(quick_text)
        return quick_start

    def create_parameters_tab(self) -> QWidget:
        """Create the Parameters tab"""
        params = QWidget()
        params_layout = QVBoxLayout(params)
        params_text = QTextBrowser()
        params_text.setHtml("""
            <h2>Analysis Parameters</h2>

            <h3>Detection Parameters:</h3>
            <ul>
                <li><b>Min Sigma:</b> Minimum particle size (pixels)
                    <br>• Typical range: 1.0 - 2.0
                    <br>• Increase for larger particles
                    <br>• Decrease for smaller particles
                    <br>• Too small: more false positives
                    <br>• Too large: might miss small particles
                </li>
                <li><b>Max Sigma:</b> Maximum particle size (pixels)
                    <br>• Typical range: 3.0 - 5.0
                    <br>• Should be larger than typical particle size
                    <br>• Affects Gaussian fitting range
                    <br>• Too small: might truncate large particles
                    <br>• Too large: slower processing
                </li>
                <li><b>Threshold:</b> Detection sensitivity (0-1)
                    <br>• Typical range: 0.1 - 0.3
                    <br>• Lower values detect more particles
                    <br>• Higher values reduce false positives
                    <br>• Start high and decrease if missing particles
                </li>
            </ul>

            <h3>Tracking Parameters:</h3>
            <ul>
                <li><b>Max Distance:</b> Maximum movement between frames
                    <br>• Should be larger than typical particle movement
                    <br>• Too small: breaks tracks unnecessarily
                    <br>• Too large: may link wrong particles
                    <br>• Consider your frame rate and particle velocity
                </li>
                <li><b>Max Gap:</b> Maximum frames a particle can disappear
                    <br>• Typical range: 1 - 3 frames
                    <br>• Helps track through missed detections
                    <br>• Increase if particles frequently disappear
                    <br>• Too large may create false connections
                </li>
                <li><b>Min Length:</b> Minimum track length
                    <br>• Filters out short tracks
                    <br>• Higher values give more reliable analysis
                    <br>• Lower values keep more tracks
                    <br>• Consider your biological timescale
                </li>
            </ul>

            <h3>Parameter Optimization:</h3>
            <ol>
                <li>Start with default parameters</li>
                <li>Use a small ROI for testing</li>
                <li>Adjust detection parameters first:
                    <ul>
                        <li>Verify detections using particle overlay</li>
                        <li>Aim for minimal false positives/negatives</li>
                    </ul>
                </li>
                <li>Then adjust tracking parameters:
                    <ul>
                        <li>Check track continuity</li>
                        <li>Verify track assignments</li>
                    </ul>
                </li>
                <li>Save working parameters for similar data</li>
            </ol>
        """)
        params_layout.addWidget(params_text)
        return params

    def create_troubleshooting_tab(self) -> QWidget:
        """Create the Troubleshooting tab"""
        trouble = QWidget()
        trouble_layout = QVBoxLayout(trouble)
        trouble_text = QTextBrowser()
        trouble_text.setHtml("""
            <h2>Troubleshooting Guide</h2>

            <h3>Detection Issues:</h3>
            <ul>
                <li><b>No particles detected:</b>
                    <ul>
                        <li>Decrease detection threshold</li>
                        <li>Verify particle size parameters</li>
                        <li>Check image contrast and brightness</li>
                        <li>Ensure correct image format</li>
                    </ul>
                </li>
                <li><b>Too many false detections:</b>
                    <ul>
                        <li>Increase detection threshold</li>
                        <li>Adjust sigma range</li>
                        <li>Use ROI to exclude noisy regions</li>
                        <li>Check for image artifacts</li>
                    </ul>
                </li>
                <li><b>Missing obvious particles:</b>
                    <ul>
                        <li>Decrease threshold</li>
                        <li>Adjust min/max sigma</li>
                        <li>Check if particles are near edges</li>
                    </ul>
                </li>
            </ul>

            <h3>Tracking Issues:</h3>
            <ul>
                <li><b>Broken tracks:</b>
                    <ul>
                        <li>Increase max gap</li>
                        <li>Increase max distance</li>
                        <li>Verify particle detection quality</li>
                        <li>Check for missed detections</li>
                    </ul>
                </li>
                <li><b>Wrong connections:</b>
                    <ul>
                        <li>Decrease max distance</li>
                        <li>Decrease max gap</li>
                        <li>Improve detection accuracy</li>
                    </ul>
                </li>
                <li><b>Short tracks:</b>
                    <ul>
                        <li>Decrease min length requirement</li>
                        <li>Improve detection consistency</li>
                        <li>Check image quality</li>
                    </ul>
                </li>
            </ul>

            <h3>Performance Issues:</h3>
            <ul>
                <li><b>Slow processing:</b>
                    <ul>
                        <li>Use smaller ROI for testing</li>
                        <li>Reduce max sigma</li>
                        <li>Process fewer frames initially</li>
                        <li>Check available memory</li>
                    </ul>
                </li>
                <li><b>Program not responding:</b>
                    <ul>
                        <li>Reduce data size</li>
                        <li>Close other applications</li>
                        <li>Check system resources</li>
                        <li>Use batch processing for large datasets</li>
                    </ul>
                </li>
            </ul>

            <h3>Analysis Issues:</h3>
            <ul>
                <li><b>Unexpected results:</b>
                    <ul>
                        <li>Verify tracking accuracy</li>
                        <li>Check parameter settings</li>
                        <li>Validate with known samples</li>
                        <li>Compare with manual analysis</li>
                    </ul>
                </li>
                <li><b>Export problems:</b>
                    <ul>
                        <li>Check file permissions</li>
                        <li>Verify disk space</li>
                        <li>Try different export format</li>
                    </ul>
                </li>
            </ul>
        """)
        trouble_layout.addWidget(trouble_text)
        return trouble

    def create_documentation_tab(self) -> QWidget:
        """Create the Documentation tab"""
        docs = QWidget()
        docs_layout = QVBoxLayout(docs)
        docs_text = QTextBrowser()
        docs_text.setOpenExternalLinks(True)

        # Create documentation links with proper paths
        doc_links = {
            'User Guide': 'user_guide.md',
            'Installation Guide': 'installation_guide.md',
            'Development Guide': 'development_guide.md',
            'GPU Optimization': 'specialized/gpu_optimization.md',
            'Advanced Analysis': 'specialized/advanced_analysis.md',
            'Batch Processing': 'specialized/batch_processing.md'
        }

        html_content = "<h2>Documentation</h2><p>Full documentation is available:</p><ul>"

        for doc_name, doc_file in doc_links.items():
            doc_path = self.docs_dir / doc_file
            if doc_path.exists():
                href = f"file:///{doc_path.as_posix()}"
                html_content += f'<li><a href="{href}">{doc_name}</a></li>'
            else:
                html_content += f'<li>{doc_name} (not available)</li>'

        html_content += """
        </ul>

        <h3>Additional Resources:</h3>
        <ul>
            <li><b>Source Code:</b> Available on GitHub</li>
            <li><b>Issue Tracker:</b> Report bugs and request features</li>
            <li><b>Examples:</b> Sample data and analysis scripts</li>
        </ul>

        <h3>Getting Help:</h3>
        <ul>
            <li>Check the troubleshooting guide</li>
            <li>Search existing issues on GitHub</li>
            <li>Report new issues with:
                <ul>
                    <li>Description of the problem</li>
                    <li>Steps to reproduce</li>
                    <li>Expected vs actual behavior</li>
                    <li>System information</li>
                </ul>
            </li>
        </ul>
        """

        docs_text.setHtml(html_content)
        docs_layout.addWidget(docs_text)
        return docs

    def report_issue(self):
        """Open issue reporting link"""
        QDesktopServices.openUrl(QUrl("https://github.com/gddickinson/particle_analysis/issues"))
