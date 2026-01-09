#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

"""
反射強度（Remission）とGround Truth（セマンティックラベル）を同時に可視化するツール

使い方:
  python visualize_remission.py --scan_path <velodyne_path> --label_path <labels_path> --config <config.yaml>

キー操作:
  n: 次のスキャン
  b: 前のスキャン
  q: 終了
"""

import argparse
import os
import yaml
import math
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.vispy_manager import VispyManager
import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.visuals.transforms import STTransform
import numpy as np
from matplotlib import pyplot as plt


class ColorbarOverlay:
    """反射強度のカラーバーを表示するオーバーレイ"""

    def __init__(self, canvas, cmap_name='viridis', white_background=False):
        self.canvas = canvas
        self.cmap_name = cmap_name
        self.white_background = white_background
        self.root = vispy.scene.Node(parent=self.canvas.scene) if self.canvas else None
        if self.root:
            self.root.transform = STTransform()
        
        self.font_size = 8
        self.margin = 8
        self.bar_width = 15
        self.bar_height = 100
        self.text_color = (0, 0, 0, 0.85) if white_background else (1, 1, 1, 0.9)
        
        if self.canvas:
            self._build()

    def _build(self):
        # カラーバーの位置（右上）
        x = self.margin
        y = self.margin
        
        # カラーバーのグラデーション用に複数の矩形を描画
        n_steps = 50
        step_height = self.bar_height / n_steps
        cmap = plt.get_cmap(self.cmap_name)
        
        for i in range(n_steps):
            value = 1.0 - i / n_steps  # 上が高い値
            color = cmap(value)
            rect = visuals.Rectangle(
                center=(x + self.bar_width / 2, y + i * step_height + step_height / 2),
                width=self.bar_width,
                height=step_height + 1,  # 隙間を埋める
                color=color,
                border_color=(0, 0, 0, 0),
                parent=self.root
            )
            rect.transform = self.root.transform
            rect.set_gl_state(depth_test=False)
            rect.order = 1

        # ラベル
        high_text = visuals.Text(
            "High",
            color=self.text_color,
            parent=self.root,
            font_size=self.font_size,
            anchor_x='left',
            anchor_y='center'
        )
        high_text.pos = (x + self.bar_width + 4, y + 5)
        high_text.transform = self.root.transform
        high_text.order = 2

        low_text = visuals.Text(
            "Low",
            color=self.text_color,
            parent=self.root,
            font_size=self.font_size,
            anchor_x='left',
            anchor_y='center'
        )
        low_text.pos = (x + self.bar_width + 4, y + self.bar_height - 5)
        low_text.transform = self.root.transform
        low_text.order = 2

        title_text = visuals.Text(
            "Remission",
            color=self.text_color,
            parent=self.root,
            font_size=self.font_size,
            anchor_x='center',
            anchor_y='bottom'
        )
        title_text.pos = (x + self.bar_width / 2, y + self.bar_height + 10)
        title_text.transform = self.root.transform
        title_text.order = 2


class LegendOverlay:
    """セマンティックラベルの凡例を表示するオーバーレイ"""

    def __init__(self, canvas, entries, white_background=False, y_offset=0):
        self.canvas = canvas
        self.entries = entries
        self.white_background = white_background
        self.y_offset = y_offset  # カラーバーの下に配置するためのオフセット
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
        # y_offsetを適用してカラーバーと重ならないようにする
        bg_center = (self.margin + bg_width / 2, self.margin + self.y_offset + bg_height / 2)

        self.background = visuals.Rectangle(
            center=bg_center,
            width=bg_width,
            height=bg_height,
            radius=2,
            color=self.bg_color,
            border_color=self.border_color,
            parent=self.root
        )
        self.background.transform = self.root.transform
        self.background.set_gl_state(depth_test=False)
        self.background.order = 1

        start_x = self.margin + self.padding
        start_y = self.margin + self.y_offset + self.padding + row_height / 2

        for idx, (label, color) in enumerate(self.entries):
            col = idx // rows
            row = idx % rows
            x = start_x + col * col_width
            y = start_y + row * row_height
            patch_color = list(color)
            if len(patch_color) == 3:
                patch_color.append(1.0)
            patch = visuals.Rectangle(
                center=(x + self.swatch_size / 2, y),
                width=self.swatch_size,
                height=self.swatch_size,
                radius=1,
                color=patch_color,
                border_color=(0, 0, 0, 0),
                parent=self.root
            )
            patch.transform = self.root.transform
            patch.set_gl_state(depth_test=False)
            patch.order = 2
            text = visuals.Text(
                label,
                color=self.text_color,
                parent=self.root,
                font_size=self.font_size,
                anchor_x='left',
                anchor_y='center'
            )
            text.pos = (x + self.swatch_size + 4, y)
            text.transform = self.root.transform
            text.order = 3
            self.items.append((patch, text))


class RemissionGTVis(VispyManager):
    """反射強度とGTを同時に可視化するクラス"""

    def __init__(self, scan, sem_scan, scan_names, label_names, offset=0, 
                 images=True, link=True, white_background=False, 
                 legend_entries=None, cmap='viridis'):
        super().__init__(offset, len(scan_names), images, instances=False, 
                        white_background=white_background)
        self.scan = scan  # 反射強度用（LaserScan）
        self.sem_scan = sem_scan  # GT用（SemLaserScan）
        self.scan_names = scan_names
        self.label_names = label_names
        self.link = link
        self.legend_entries = legend_entries or []
        self.cmap_name = cmap
        
        # Viewとビジュアルの初期化
        self.remission_view = None
        self.remission_vis = None
        self.gt_view = None
        self.gt_vis = None
        self.img_remission_view = None
        self.img_remission_vis = None
        self.img_gt_view = None
        self.img_gt_vis = None
        
        # オーバーレイ
        self.colorbar_overlay = None
        self.legend_overlay = None
        
        self.reset()
        self._setup_overlays()
        self.update_scan()

    def get_mpl_colormap(self, cmap_name):
        """matplotlibのカラーマップを取得"""
        cmap = plt.get_cmap(cmap_name)
        sm = plt.cm.ScalarMappable(cmap=cmap)
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def reset(self):
        """Viewboxを設定"""
        # 上: 反射強度（row=0）
        self.remission_view, self.remission_vis = super().add_viewbox(0, 0)
        
        # 下: GT（row=1）
        self.gt_view, self.gt_vis = super().add_viewbox(1, 0)
        
        # カメラをリンク
        if self.link:
            self.remission_view.camera.link(self.gt_view.camera)
        
        # 画像ビュー
        if self.images:
            self.img_remission_view, self.img_remission_vis = super().add_image_viewbox(0, 0)
            self.img_gt_view, self.img_gt_vis = super().add_image_viewbox(1, 0)

    def _setup_overlays(self):
        """オーバーレイを設定"""
        # カラーバー（反射強度用）- 左上
        self.colorbar_overlay = ColorbarOverlay(
            self.canvas, 
            cmap_name=self.cmap_name,
            white_background=self.white_background
        )
        
        # 凡例（GT用）- カラーバーの下に配置
        # カラーバーの高さ: bar_height(100) + margin(8) + title(~25) = 約135
        if self.legend_entries:
            self.legend_overlay = LegendOverlay(
                self.canvas,
                self.legend_entries,
                white_background=self.white_background,
                y_offset=140  # カラーバーの下に配置
            )

    def update_scan(self):
        """スキャンを更新"""
        # 反射強度用スキャンを開く
        self.scan.open_scan(self.scan_names[self.offset])
        
        # GT用スキャンを開く
        self.sem_scan.open_scan(self.scan_names[self.offset])
        self.sem_scan.open_label(self.label_names[self.offset])
        self.sem_scan.colorize()
        
        # === 反射強度の可視化 ===
        remissions = np.copy(self.scan.remissions)
        
        # 正規化（0-255）
        if remissions.max() > remissions.min():
            remissions_normalized = ((remissions - remissions.min()) / 
                                     (remissions.max() - remissions.min()) * 255).astype(np.uint8)
        else:
            remissions_normalized = np.zeros_like(remissions, dtype=np.uint8)
        
        # カラーマップ適用
        viridis_map = self.get_mpl_colormap(self.cmap_name)
        remission_colors = viridis_map[remissions_normalized]
        
        self.remission_vis.set_data(
            self.scan.points,
            face_color=remission_colors[..., ::-1],
            edge_color=remission_colors[..., ::-1],
            size=1
        )
        
        # === GTの可視化 ===
        self.gt_vis.set_data(
            self.sem_scan.points,
            face_color=self.sem_scan.sem_label_color[..., ::-1],
            edge_color=self.sem_scan.sem_label_color[..., ::-1],
            size=1
        )
        
        # === 画像ビューの更新 ===
        if self.images:
            # 反射強度の投影画像
            proj_remission = np.copy(self.scan.proj_remission)
            proj_remission[proj_remission < 0] = 0  # 無効値を0に
            if proj_remission.max() > proj_remission.min():
                proj_remission = (proj_remission - proj_remission.min()) / \
                                (proj_remission.max() - proj_remission.min())
            self.img_remission_vis.set_data(proj_remission)
            self.img_remission_vis.update()
            
            # GTの投影画像
            self.img_gt_vis.set_data(self.sem_scan.proj_sem_color[..., ::-1])
            self.img_gt_vis.update()
        
        # タイトル更新
        title = f"scan {self.offset} (top: Remission, bottom: GT)"
        self.canvas.title = title
        if self.images:
            self.img_canvas.title = title


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./visualize_remission.py")
    parser.add_argument(
        '--config_run',
        type=str,
        required=False,
        default=None,
        help='YAML file with run configuration.',
    )
    parser.add_argument(
        '--scan_path', '-s',
        type=str,
        required=False,
        help='Path to point cloud scans (velodyne folder)',
    )
    parser.add_argument(
        '--label_path', '-l',
        type=str,
        required=False,
        help='Path to label folder (labels folder)',
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
        action='store_true',
        help='Do not visualize range image projections.',
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Sequence to start. Defaults to %(default)s',
    )
    parser.add_argument(
        '--white_background',
        dest='white_background',
        default=False,
        action='store_true',
        help='Use white background instead of black.',
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='Colormap for remission visualization. Defaults to %(default)s',
    )
    parser.add_argument(
        '--color_learning_map',
        dest='color_learning_map',
        default=False,
        action='store_true',
        help='Apply learning map to color map for GT.',
    )

    # First parse to get config_run
    FLAGS_temp, _ = parser.parse_known_args()

    # Load run config if specified
    if FLAGS_temp.config_run:
        try:
            print(f"Loading run config file {FLAGS_temp.config_run}")
            run_config = yaml.safe_load(open(FLAGS_temp.config_run, 'r'))
            parser.set_defaults(**run_config)
        except Exception as e:
            print(e)
            print("Error opening run config yaml file.")
            quit()

    FLAGS, _ = parser.parse_known_args()

    # Validate required arguments
    if not FLAGS.scan_path:
        print("Error: scan_path must be specified")
        quit()
    if not FLAGS.label_path:
        print("Error: label_path must be specified")
        quit()

    # Print summary
    print("*" * 80)
    print("REMISSION + GT VISUALIZER")
    print("*" * 80)
    print(f"Scan path: {FLAGS.scan_path}")
    print(f"Label path: {FLAGS.label_path}")
    print(f"Config: {FLAGS.config}")
    print(f"ignore_images: {FLAGS.ignore_images}")
    print(f"offset: {FLAGS.offset}")
    print(f"white_background: {FLAGS.white_background}")
    print(f"cmap: {FLAGS.cmap}")
    print(f"color_learning_map: {FLAGS.color_learning_map}")
    print("*" * 80)

    # Open config file
    try:
        print(f"Opening config file {FLAGS.config}")
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # Check scan path
    if not os.path.isdir(FLAGS.scan_path):
        print(f"Scan folder {FLAGS.scan_path} doesn't exist! Exiting...")
        quit()

    # Check label path
    if not os.path.isdir(FLAGS.label_path):
        print(f"Label folder {FLAGS.label_path} doesn't exist! Exiting...")
        quit()

    # Get scan names
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(FLAGS.scan_path)) for f in fn]
    scan_names.sort()
    print(f"Found {len(scan_names)} scans")

    # Get label names
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(FLAGS.label_path)) for f in fn]
    label_names.sort()
    print(f"Found {len(label_names)} labels")

    # Align scans and labels
    scan_map = {os.path.splitext(os.path.basename(f))[0]: f for f in scan_names}
    label_map = {os.path.splitext(os.path.basename(f))[0]: f for f in label_names}
    common_keys = set(scan_map.keys()) & set(label_map.keys())

    if not common_keys:
        print("No common scan ids found. Exiting...")
        quit()

    def sort_key(key):
        return int(key) if key.isdigit() else key

    sorted_keys = sorted(common_keys, key=sort_key)
    print(f"Using {len(sorted_keys)} aligned scans/labels")

    scan_names = [scan_map[k] for k in sorted_keys]
    label_names = [label_map[k] for k in sorted_keys]

    # Reset offset if needed
    if FLAGS.offset >= len(scan_names):
        print(f"Requested offset {FLAGS.offset} exceeds available scans ({len(scan_names)}). Resetting to 0.")
        FLAGS.offset = 0

    # Setup color map
    base_color_map = CFG["color_map"]
    label_dict = CFG.get("labels", {})
    legend_entries = []

    if FLAGS.color_learning_map:
        learning_map_inv = CFG["learning_map_inv"]
        learning_map = CFG["learning_map"]
        color_dict = {key: base_color_map[learning_map_inv[learning_map[key]]]
                      for key in base_color_map.keys()}
        for learn_id in sorted(learning_map_inv.keys()):
            rep_id = learning_map_inv[learn_id]
            color = base_color_map[rep_id]
            label_text = label_dict.get(rep_id, str(rep_id))
            legend_entries.append((label_text, [c / 255.0 for c in color[::-1]]))
    else:
        color_dict = base_color_map
        for class_id, color in sorted(color_dict.items()):
            label_text = label_dict.get(class_id, str(class_id))
            legend_entries.append((label_text, [c / 255.0 for c in color[::-1]]))

    # Create scans
    scan = LaserScan(project=True)  # 反射強度用
    sem_scan = SemLaserScan(color_dict, project=True)  # GT用

    # Create visualizer
    images = not FLAGS.ignore_images
    vis = RemissionGTVis(
        scan=scan,
        sem_scan=sem_scan,
        scan_names=scan_names,
        label_names=label_names,
        offset=FLAGS.offset,
        images=images,
        link=True,
        white_background=FLAGS.white_background,
        legend_entries=legend_entries,
        cmap=FLAGS.cmap
    )

    # Print instructions
    print("\nTo navigate:")
    print("\tn: next (next scan)")
    print("\tb: back (previous scan)")
    print("\tq: quit (exit program)")

    # Run visualizer
    vis.run()

