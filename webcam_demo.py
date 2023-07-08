import torch
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--img_width', type=int, default=2560) # 2560, 1920
parser.add_argument('--img_height', type=int, default=1440) # 1440, 1080
parser.add_argument('--scale_factor', type=float, default=0.7125) # 0.2 for higher fps
parser.add_argument('--line_thickness', type=int, default=6)
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.to('cpu')

    output_stride = model.output_stride

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count, curr_frames = 0, 0
    while True:
        curr_start = time.time()
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).to('cpu')

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        num_people = len([score for score in pose_scores if score > 0.15])
        avg_confidence = sum(pose_scores) / len(pose_scores) if len(pose_scores) > 0 else 0

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords, line_thickness=args.line_thickness,
            min_pose_score=0.15, min_part_score=0.1)

        overlay = overlay_image.copy()
        overlay_resized = cv2.resize(overlay, dsize=(args.img_width, args.img_height))
        display_image_resized = cv2.resize(display_image, dsize=(args.img_width, args.img_height))
        frame_count += 1
        fps = 1 / (time.time() - curr_start)
        cv2.rectangle(overlay_resized, (0, 0), (430, 130), (0, 255, 0), -1)  # Draw a green rectangle
        cv2.addWeighted(overlay_resized, 0.5, display_image_resized, 0.5, 0, display_image_resized)
        cv2.putText(display_image_resized, f"Number of people: {num_people}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_image_resized, f"Average confidence: {avg_confidence:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_image_resized, f"FPS: {fps:.2f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('posenet', display_image_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()