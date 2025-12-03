#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import sys
import yaml
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis
from auxiliary.vispy_manager import VispyManager
import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np

class LaserScanMultiComp(VispyManager):
  """Class that creates and handles a multi-view pointcloud comparison"""

  def __init__(self, scans, scan_names, label_names, offset=0, images=True, instances=False, link=False, split_direction='horizontal', white_background=False):
    super().__init__(offset, len(scan_names), images, instances, white_background)
    self.scans = scans
    self.scan_names = scan_names
    self.label_names = label_names
    self.link = link
    self.split_direction = split_direction  # 'horizontal' or 'vertical'
    self.views = []
    self.visuals = []
    self.img_views = []
    self.img_visuals = []
    self.inst_views = []
    self.inst_visuals = []
    self.img_inst_views = []
    self.img_inst_visuals = []
    self.reset()
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
  color_dict = CFG["color_map"]
  if FLAGS.color_learning_map:
    learning_map_inv = CFG["learning_map_inv"]
    learning_map = CFG["learning_map"]
    color_dict = {key: color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

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
                     white_background=FLAGS.white_background)

  # print instructions
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")

  # run the visualizer
  vis.run()
