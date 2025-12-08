#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import sys
import yaml
import math
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis
from auxiliary.vispy_manager import VispyManager
import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.visuals.transforms import STTransform
import numpy as np

class LegendOverlay:
  """Small, unobtrusive legend showing color-to-label mapping."""

  def __init__(self, canvas, entries, white_background=False):
    self.canvas = canvas
    self.entries = entries
    self.white_background = white_background
    self.root = vispy.scene.Node(parent=self.canvas.scene) if self.canvas else None
    if self.root:
      self.root.transform = STTransform()
    self.font_size = 8
    self.margin = 8
    self.padding = 6
    self.max_rows = 12
    self.swatch_size = 10
    self.text_color = (0, 0, 0, 0.85) if white_background else (1, 1, 1, 0.9)
    self.bg_color = (0, 0, 0, 0.08) if white_background else (0, 0, 0, 0.4)
    self.border_color = (0, 0, 0, 0.15) if white_background else (1, 1, 1, 0.15)
    self.items = []
    self.background = None
    if self.canvas and self.entries:
      self._build()

  def _approx_text_width(self, text_len):
    # crude width estimation to size columns without depending on font metrics
    return max(int(text_len * self.font_size * 0.6), self.font_size)

  def _build(self):
    rows = min(self.max_rows, len(self.entries))
    cols = int(math.ceil(len(self.entries) / rows))
    longest_label = max(len(label) for label, _ in self.entries)
    col_width = self.swatch_size + 6 + self._approx_text_width(longest_label)
    row_height = max(self.swatch_size, int(self.font_size * 1.3))

    legend_width = cols * col_width
    legend_height = rows * row_height

    bg_width = legend_width + self.padding * 2
    bg_height = legend_height + self.padding * 2
    bg_center = (self.margin + bg_width / 2, self.margin + bg_height / 2)

    self.background = visuals.Rectangle(center=bg_center,
                                        width=bg_width,
                                        height=bg_height,
                                        radius=2,
                                        color=self.bg_color,
                                        border_color=self.border_color,
                                        parent=self.root)
    self.background.transform = self.root.transform
    self.background.set_gl_state(depth_test=False)
    self.background.order = 1

    start_x = self.margin + self.padding
    start_y = self.margin + self.padding + row_height / 2

    for idx, (label, color) in enumerate(self.entries):
      col = idx // rows
      row = idx % rows
      x = start_x + col * col_width
      y = start_y + row * row_height
      patch_color = list(color)
      if len(patch_color) == 3:
        patch_color.append(1.0)
      patch = visuals.Rectangle(center=(x + self.swatch_size / 2, y),
                                width=self.swatch_size,
                                height=self.swatch_size,
                                radius=1,
                                color=patch_color,
                                border_color=(0, 0, 0, 0),
                                parent=self.root)
      patch.transform = self.root.transform
      patch.set_gl_state(depth_test=False)
      patch.order = 2
      text = visuals.Text(label,
                          color=self.text_color,
                          parent=self.root,
                          font_size=self.font_size,
                          anchor_x='left',
                          anchor_y='center')
      text.pos = (x + self.swatch_size + 4, y)
      text.transform = self.root.transform
      text.order = 3
      self.items.append((patch, text))

class LaserScanMultiComp(VispyManager):
  """Class that creates and handles a multi-view pointcloud comparison"""

  def __init__(self, scans, scan_names, label_names, offset=0, images=True, instances=False, link=False, split_direction='horizontal', white_background=False, legend_entries=None):
    super().__init__(offset, len(scan_names), images, instances, white_background)
    self.scans = scans
    self.scan_names = scan_names
    self.label_names = label_names
    self.link = link
    self.split_direction = split_direction  # 'horizontal' or 'vertical'
    self.legend_entries = legend_entries or []
    self.legend_overlay = None
    self.img_legend_overlay = None
    self.views = []
    self.visuals = []
    self.img_views = []
    self.img_visuals = []
    self.inst_views = []
    self.inst_visuals = []
    self.img_inst_views = []
    self.img_inst_visuals = []
    self.reset()
    self._setup_legends()
    self.update_scan()

  def reset(self):
    """prepares the canvas(es) for the visualizer"""
    n_scans = len(self.scans)
    
    # Create views for each scan based on split direction
    for i in range(n_scans):
      if self.split_direction == 'horizontal':
        # Horizontal split (side by side)
        row, col = 0, i
      else:
        # Vertical split (top to bottom)
        row, col = i, 0
        
      view, vis = super().add_viewbox(row, col)
      self.views.append(view)
      self.visuals.append(vis)
      
      # Link cameras if requested
      if self.link and i > 0:
        self.views[0].camera.link(view.camera)
    
    # Add image views if requested
    if self.images:
      for i in range(n_scans):
        img_view, img_vis = super().add_image_viewbox(i, 0)
        self.img_views.append(img_view)
        self.img_visuals.append(img_vis)
        
        if self.instances:
          img_inst_view, img_inst_vis = super().add_image_viewbox(i + n_scans, 0)
          self.img_inst_views.append(img_inst_view)
          self.img_inst_visuals.append(img_inst_vis)
    
    # Add instance views if requested
    if self.instances:
      for i in range(n_scans):
        if self.split_direction == 'horizontal':
          row, col = 1, i
        else:
          row, col = i + n_scans, 0
          
        inst_view, inst_vis = super().add_viewbox(row, col)
        self.inst_views.append(inst_view)
        self.inst_visuals.append(inst_vis)
        
        # Link cameras if requested
        if self.link:
          self.views[i].camera.link(inst_view.camera)

  def update_scan(self):
    """updates the scans, images and instances"""
    for i, scan in enumerate(self.scans):
      scan.open_scan(self.scan_names[self.offset])
      scan.open_label(self.label_names[i][self.offset])
      scan.colorize()
      self.visuals[i].set_data(scan.points,
                            face_color=scan.sem_label_color[..., ::-1],
                            edge_color=scan.sem_label_color[..., ::-1],
                            size=1)

    if self.instances:
      for i, scan in enumerate(self.scans):
        self.inst_visuals[i].set_data(scan.points,
                                 face_color=scan.inst_label_color[..., ::-1],
                                 edge_color=scan.inst_label_color[..., ::-1],
                                 size=1)

    if self.images:
      for i, scan in enumerate(self.scans):
        self.img_visuals[i].set_data(scan.proj_sem_color[..., ::-1])
        self.img_visuals[i].update()

        if self.instances:
          self.img_inst_visuals[i].set_data(scan.proj_inst_color[..., ::-1])
          self.img_inst_visuals[i].update()
    
    # Update window title with current scan number
    title = "scan " + str(self.offset)
    self.canvas.title = title
    if self.images:
      self.img_canvas.title = title

  def _setup_legends(self):
    if not self.legend_entries:
      return
    self.legend_overlay = LegendOverlay(self.canvas,
                                        self.legend_entries,
                                        white_background=self.white_background)
    if self.images and self.img_canvas:
      self.img_legend_overlay = LegendOverlay(self.img_canvas,
                                              self.legend_entries,
                                              white_background=self.white_background)

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./compare.py")
  parser.add_argument(
      '--config_run',
      type=str,
      required=False,
      default=None,
      help='YAML file with run configuration. If specified, values from this file are used as defaults.',
  )
  parser.add_argument(
      '--scan_path', '-s',
      type=str,
      required=False,
      help='Path to point cloud scans. No Default',
  )
  parser.add_argument(
      '--label_paths', '-l',
      required=False,
      nargs='+',
      help='Paths to label folders to visualize. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_images', '-r',
      dest='ignore_images',
      default=False,
      required=False,
      action='store_true',
      help='Do not visualize range image projections. Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-i',
      dest='do_instances',
      default=False,
      required=False,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--link', '-k',
      dest='link',
      default=False,
      required=False,
      action='store_true',
      help='Link viewpoint changes across windows. Defaults to %(default)s',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      required=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
    '--color_learning_map',
    dest='color_learning_map',
    default=False,
    required=False,
    action='store_true',
    help='Apply learning map to color map: visualize only classes that were trained on',
  )
  parser.add_argument(
    '--split_direction',
    type=str,
    default='horizontal',
    choices=['horizontal', 'vertical'],
    required=False,
    help='Direction to split visualization windows. Defaults to %(default)s',
  )
  parser.add_argument(
    '--random_colors',
    type=str,
    nargs='+',
    default=[],
    required=False,
    help='List of label path indices (0-based) to use random colors instead of semantic colors',
  )
  parser.add_argument(
    '--random_seed',
    type=int,
    default=42,
    required=False,
    help='Random seed for generating random colors. Defaults to %(default)s',
  )
  parser.add_argument(
    '--max_label',
    type=int,
    default=20000,
    required=False,
    help='Maximum label value for random colors. Defaults to %(default)s',
  )
  parser.add_argument(
    '--white_background',
    dest='white_background',
    default=False,
    required=False,
    action='store_true',
    help='Use white background instead of black. Defaults to %(default)s',
  )
  # First, parse only to get config_run
  FLAGS_temp, _ = parser.parse_known_args()
  
  # Load run config if specified and set as defaults
  if FLAGS_temp.config_run:
    try:
      print("Loading run config file %s" % FLAGS_temp.config_run)
      run_config = yaml.safe_load(open(FLAGS_temp.config_run, 'r'))
      # Set defaults from YAML config
      parser.set_defaults(**run_config)
    except Exception as e:
      print(e)
      print("Error opening run config yaml file.")
      quit()
  
  # Parse again with updated defaults (command line args will override defaults)
  FLAGS, unparsed = parser.parse_known_args()

  # Validate required arguments
  if not FLAGS.scan_path:
    print("Error: scan_path must be specified either via command line or config file")
    quit()
  if not FLAGS.label_paths:
    print("Error: label_paths must be specified either via command line or config file")
    quit()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Scan path: ", FLAGS.scan_path)
  print("Label paths: ", FLAGS.label_paths)
  print("Config", FLAGS.config)
  print("ignore_images", FLAGS.ignore_images)
  print("do_instances", FLAGS.do_instances)
  print("link", FLAGS.link)
  print("ignore_safety", FLAGS.ignore_safety)
  print("color_learning_map", FLAGS.color_learning_map)
  print("offset", FLAGS.offset)
  print("split_direction", FLAGS.split_direction)
  print("random_colors", FLAGS.random_colors)
  print("random_seed", FLAGS.random_seed)
  print("max_label", FLAGS.max_label)
  print("white_background", FLAGS.white_background)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # does scan path exist?
  scan_path = FLAGS.scan_path
  if os.path.isdir(scan_path):
    print("Scan folder exists! Using scans from %s" % scan_path)
  else:
    print(f"Scan folder {scan_path} doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_path)) for f in fn]
  scan_names.sort()

  print(f"Found {len(scan_names)} scans")

  # check if label paths exist
  label_names = []
  for i, label_path in enumerate(FLAGS.label_paths):
    if os.path.isdir(label_path):
      print(f"Labels folder {i+1} exists! Using labels from {label_path}")
    else:
      print(f"Labels folder {label_path} doesn't exist! Exiting...")
      quit()
      
    # populate the labels
    labels = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_path)) for f in fn]
    labels.sort()
    label_names.append(labels)
    
    # check that there are same amount of labels and scans
    if not FLAGS.ignore_safety:
      assert len(labels) == len(scan_names), f"Number of labels in {label_path} ({len(labels)}) doesn't match number of scans ({len(scan_names)})"

  # create scans
  base_color_map = CFG["color_map"]
  label_dict = CFG.get("labels", {})
  legend_entries = []

  if FLAGS.color_learning_map:
    learning_map_inv = CFG["learning_map_inv"]
    learning_map = CFG["learning_map"]
    # Reassign colors so that classes mapped together share the representative color
    color_dict = {key: base_color_map[learning_map_inv[learning_map[key]]]
                  for key in base_color_map.keys()}
    # Build legend on learning-map (20 classes)
    for learn_id in sorted(learning_map_inv.keys()):
      rep_id = learning_map_inv[learn_id]
      color = base_color_map[rep_id]
      label_text = label_dict.get(rep_id, str(rep_id))
      legend_entries.append((label_text, [c / 255.0 for c in color[::-1]]))
  else:
    color_dict = base_color_map
    # Build legend on full set (34 classes)
    for class_id, color in sorted(color_dict.items()):
      label_text = label_dict.get(class_id, str(class_id))
      legend_entries.append((label_text, [c / 255.0 for c in color[::-1]]))

  scans = []
  for i in range(len(FLAGS.label_paths)):
    # 指定されたインデックスのラベルパスにはランダムな色を使用
    if str(i) in FLAGS.random_colors:
      print(f"Using random colors for label path {i+1} with seed {FLAGS.random_seed} and max_label {FLAGS.max_label}")
      # 再現性のためにシードを設定
      random_state = np.random.RandomState(FLAGS.random_seed)
      
      # ユニークなラベル値を取得するための仮のカラーマップを作成
      random_color_dict = {}
      for j in range(int(FLAGS.max_label)):
        random_color_dict[j] = random_state.randint(0, 256, 3).tolist()
      
      scans.append(SemLaserScan(random_color_dict, project=True, random_seed=FLAGS.random_seed))
    else:
      # 通常のセマンティックカラーを使用
      scans.append(SemLaserScan(color_dict, project=True, random_seed=FLAGS.random_seed))

  # create a visualizer
  images = not FLAGS.ignore_images
  vis = LaserScanMultiComp(scans=scans,
                     scan_names=scan_names,
                     label_names=label_names,
                     offset=FLAGS.offset, 
                     images=images, 
                     instances=FLAGS.do_instances, 
                     link=FLAGS.link,
                     split_direction=FLAGS.split_direction,
                     white_background=FLAGS.white_background,
                     legend_entries=legend_entries)

  # print instructions
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")

  # run the visualizer
  vis.run()
