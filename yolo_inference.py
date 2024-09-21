from ultralytics import YOLO

# Load the YOLO model from the given file path
model = YOLO('models/best.pt')

# Use the model to predict objects in the input video, saving the output
results = model.predict('input_videos/08fd33_4.mp4', save=True)

# Print the first set of prediction results (for the first frame of the video)
print(results[0])

print('=======================')

# Loop through each bounding box in the first frame's results and print them
for box in results[0].boxes:
    print(box)
