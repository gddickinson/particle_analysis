# tests/test_visualization.py

[previous content remains the same until the feature viewer test...]

def test_feature_viewer(qtbot, sample_features):
    """Test FeatureViewer widget"""
    viewer = FeatureViewer()
    qtbot.addWidget(viewer)
    
    # Test feature setting
    viewer.set_features(sample_features)
    assert viewer.feature_combo.count() > 0
    
    # Test feature selection
    feature_name = viewer.feature_combo.currentText()
    assert feature_name != ""
    
    # Test display update
    viewer.update_display()
    assert len(viewer.plot_widget.plotItem.items) > 0
    
    # Test feature selection signal
    with qtbot.waitSignal(viewer.feature_selected) as blocker:
        viewer.feature_combo.setCurrentText('alpha')
    assert blocker.args == ['alpha']
    
    # Test clearing
    viewer.clear()
    assert len(viewer.plot_widget.plotItem.items) == 0

def test_visualization_error_handling(sample_tracks):
    """Test error handling in visualization"""
    visualizer = TrackVisualizer()
    
    # Test with empty track list
    ax = visualizer.plot_tracks([])
    assert ax is not None
    
    # Test with invalid feature names
    ax = visualizer.plot_tracks(sample_tracks, color_by='invalid_feature')
    assert ax is not None
    
    # Test movie creation with invalid path
    success = visualizer.create_track_movie(
        sample_tracks,
        "/invalid/path/movie.mp4"
    )
    assert not success

def test_interactive_track_selection(qtbot, sample_tracks):
    """Test interactive track selection in viewer"""
    viewer = TrackViewer()
    qtbot.addWidget(viewer)
    
    # Set up signal capture
    selected_tracks = []
    viewer.track_selected.connect(lambda x: selected_tracks.append(x))
    
    # Add tracks
    viewer.set_data(sample_tracks)
    
    # Simulate track selection
    plot_item = next(iter(viewer.track_plots.values()))
    qtbot.mouseClick(viewer.plot_widget.viewport(), Qt.MouseButton.LeftButton,
                    pos=plot_item.mapToView(plot_item.getData()[0][0]))
    
    # Check if signal was emitted
    assert len(selected_tracks) > 0

def test_feature_distribution_styling(sample_features):
    """Test visualization styling options"""
    visualizer = TrackVisualizer(colormap='plasma', dpi=150, style='seaborn')
    
    # Test custom styling
    fig = visualizer.plot_track_feature_distributions(sample_features)
    assert fig is not None
    assert fig.dpi == 150

def test_viewer_synchronization(qtbot):
    """Test synchronization between track and feature viewers"""
    track_viewer = TrackViewer()
    feature_viewer = FeatureViewer()
    qtbot.addWidget(track_viewer)
    qtbot.addWidget(feature_viewer)
    
    # Create synchronized displays
    def sync_viewers(feature_name):
        track_viewer.color_combo.setCurrentText(feature_name)
        
    feature_viewer.feature_selected.connect(sync_viewers)
    
    # Test synchronization
    with qtbot.waitSignal(track_viewer.track_selected):
        feature_viewer.feature_combo.setCurrentText('diffusion_coefficient')
    
    assert track_viewer.color_combo.currentText() == 'diffusion_coefficient'

def test_real_time_updates(qtbot, sample_tracks):
    """Test real-time visualization updates"""
    viewer = TrackViewer()
    qtbot.addWidget(viewer)
    
    # Set initial data
    viewer.set_data(sample_tracks)
    initial_plot_count = len(viewer.plot_widget.plotItem.items)
    
    # Add new track
    new_particles = [
        Particle(frame=i, x=float(i), y=float(i), intensity=100.0,
                sigma=1.5, snr=10.0, frame_size=(100, 100))
        for i in range(5)
    ]
    new_track = Track(id=3, particles=new_particles,
                     start_frame=0, end_frame=4)
    
    # Update visualization
    viewer.set_data(sample_tracks + [new_track])
    
    # Check if new track was added
    assert len(viewer.plot_widget.plotItem.items) > initial_plot_count

def test_viewer_performance(qtbot, sample_tracks):
    """Test viewer performance with large datasets"""
    viewer = TrackViewer()
    qtbot.addWidget(viewer)
    
    # Create large dataset
    large_tracks = sample_tracks * 100  # Multiply existing tracks
    
    # Measure time to update display
    import time
    start_time = time.time()
    viewer.set_data(large_tracks)
    viewer.update_display()
    end_time = time.time()
    
    # Performance should be reasonable
    assert end_time - start_time < 1.0  # Should update in less than 1 second