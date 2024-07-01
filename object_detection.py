import cv2
import numpy as np
import tensorflow as tf

# Load frozen inference graph
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')

    return graph

# Load labels from file
def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()
    category_index = {i + 1: {'id': i + 1, 'name': label} for i, label in enumerate(labels)}
    return category_index

# Run object detection on a frame
def run_inference_for_single_image(session, image_tensor, detection_boxes, detection_scores, detection_classes, image):
    print(f"Input shape: {image.shape}, Input type: {image.dtype}")
    output_dict = session.run(
        [detection_boxes, detection_scores, detection_classes],
        feed_dict={image_tensor: image})

    output_dict = {
        'detection_boxes': np.squeeze(output_dict[0]),
        'detection_scores': np.squeeze(output_dict[1]),
        'detection_classes': np.squeeze(output_dict[2]).astype(np.int32),
    }

    print(f"Boxes shape: {output_dict['detection_boxes'].shape}, Scores shape: {output_dict['detection_scores'].shape}, Classes shape: {output_dict['detection_classes'].shape}")
    return output_dict

# Draw bounding boxes on frame
def draw_boxes(image, boxes, scores, classes, category_index, min_score_thresh=0.3):
    height, width, _ = image.shape
    for i in range(len(scores)):
        if scores[i] >= min_score_thresh:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            
            # Handle missing class labels
            class_index = classes[i]
            print(f"Detected class index: {class_index}")  # Debug print
            if class_index in category_index:
                label = category_index[class_index]['name']
            else:
                label = 'Unknown'
                print(f"Unknown class index detected: {class_index}")  # Debug print
            
            cv2.putText(image, '{}: {:.2f}'.format(label, scores[i]), (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image

# Load label map
labels_path = 'labels.txt'
category_index = load_labels(labels_path)

# Load model
model_file = 'model.pb'
graph = load_graph(model_file)

with graph.as_default():
    session = tf.compat.v1.Session(graph=graph)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detected_boxes:0')
    detection_scores = graph.get_tensor_by_name('detected_scores:0')
    detection_classes = graph.get_tensor_by_name('detected_classes:0')

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 320x320 as expected by model
    resized_frame = cv2.resize(frame, (320, 320))
    input_frame = np.expand_dims(resized_frame, axis=0)

    # Ensure input frame is a numpy array with float32 type
    input_frame = input_frame.astype(np.float32)

    # Run object detection
    output_dict = run_inference_for_single_image(session, image_tensor, detection_boxes, detection_scores, detection_classes, input_frame)

    # Draw boxes
    frame_with_boxes = draw_boxes(frame, output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'], category_index, min_score_thresh=0.3)

    # Display frame
    cv2.imshow('Object Detection', frame_with_boxes)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
