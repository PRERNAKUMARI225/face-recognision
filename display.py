import face_recognition
import cv2
import pickle
import os
import datetime
import mysql.connector
from gtts import gTTS
from playsound import playsound

# Establish a connection to the MySQL database
connection = mysql.connector.connect(
    host='localhost',      # Host where your MySQL server is running
    database='mydatabase',    # Name of your database
    user='root',      # Username for MySQL
    password='A1B2C3d$'   # Password for MySQL
)

# Check if the connection is successful
if connection.is_connected():
    print('Connected to MySQL database')
else:
    print('Failed to connect to MySQL database')
    exit()

# Create a cursor object
cursor = connection.cursor()

# Function to insert face data into MySQL database
def insert_face_data(name, timestamp, face_image):
    try:
        # Insert face data into the database
        query = "INSERT INTO face_data (name, timestamp) VALUES (%s, %s)"
        values = (name, timestamp)
        cursor.execute(query, values)
        connection.commit()
        print("Inserted face data into the database successfully")

        # Save the detected face image
        cv2.imwrite(f"detected_faces/{name}_{timestamp}.jpg", face_image)

        # Speak the name of the detected person in Hindi
        tts = gTTS(text=f"{name} attendance lag gaya dhanyabaad", lang='en')
        tts.save("output.mp3")
        playsound("output.mp3")

    except mysql.connector.Error as e:
        print(f"Error inserting face data into the database: {e}")
        connection.rollback()

# Load or create known faces data
known_faces_data_file = "known_faces_data.pkl"
if os.path.exists(known_faces_data_file):
    with open(known_faces_data_file, "rb") as file:
        known_faces_data = pickle.load(file)
else:
    known_faces_data = {"encodings": [], "names": []}

# Directory where face images are stored
images_directory = "attendance_images/"

# List all files in the directory
image_files = [os.path.join(images_directory, file) for file in os.listdir(images_directory) if file.endswith(('.jpg', '.jpeg', '.png'))]

for image_path in image_files:
    # Load the image
    known_image = face_recognition.load_image_file(image_path)

    # Encode the face
    face_encodings = face_recognition.face_encodings(known_image)

    if len(face_encodings) > 0:
        known_image_encoding = face_encodings[0]

        # Extract name from the file name
        name = os.path.splitext(os.path.basename(image_path))[0]

        # Add known face encoding and name to the data
        known_faces_data["encodings"].append(known_image_encoding)
        known_faces_data["names"].append(name)
    else:
        print("No face detected in", image_path)

# Save updated known faces data
with open(known_faces_data_file, "wb") as file:
    pickle.dump(known_faces_data, file)

# Initialize variables
face_locations = []
face_encodings = []
face_names = {}
last_detection_time = {}

# Set a threshold for face recognition confidence
face_recognition_threshold = 0.60 # You can adjust this value according to your needs

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # 0 for the default webcam, you can change it according to your setup

# Function to add new face from webcam
def add_new_face_from_webcam(name):
    # Capture an image from the webcam
    ret, frame = video_capture.read()

    # Resize frame for faster processing (optional)
    small_farame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV default) to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    if len(face_encodings) > 0:
        new_face_encoding = face_encodings[0]

        # Add new face encoding and name to the data
        known_faces_data["encodings"].append(new_face_encoding)
        known_faces_data["names"].append(name)

        # Save updated known faces data
        with open(known_faces_data_file, "wb") as file:
            pickle.dump(known_faces_data, file)

        print(f"Added new face: {name}")

    else:
        print("No face detected in the captured image")

# Main loop
while True:
    # Read video frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV default) to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding, location in zip(face_encodings, face_locations):
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_faces_data["encodings"], face_encoding, tolerance=face_recognition_threshold)
        name = "Unknown"
        match_percentage = 0

        if True in matches:
            # Retrieve the name of the known face
            first_match_index = matches.index(True)
            name = known_faces_data["names"][first_match_index]

            # Calculate face recognition percentage
            face_distances = face_recognition.face_distance(known_faces_data["encodings"], face_encoding)
            match_percentage = (1 - face_distances[first_match_index]) * 100

            # Retrieve last detection time from database
            query = "SELECT timestamp FROM face_data WHERE name = %s ORDER BY timestamp DESC LIMIT 1"
            cursor.execute(query, (name,))
            last_detection_result = cursor.fetchone()

            if last_detection_result:
                last_detection_time = last_detection_result[0]
                time_difference = (datetime.datetime.now() - last_detection_time).total_seconds()

                # Check if the face has been detected in the last 15 minutes
                if time_difference >= 900:
                    if match_percentage >= (face_recognition_threshold*100):  # You can adjust this threshold according to your needs
                        # Save face data to MySQL database
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        insert_face_data(name, timestamp, frame)
                    else:
                        # If match percentage is below threshold, set name to "Unknown"
                        name = "Unknown"

        # Store whether the face is known or unknown
        face_names[name] = matches[0]

        # Display results only if the face is known
        if name != "Unknown" and match_percentage >= (face_recognition_threshold*100):
            top, right, bottom, left = [i * 4 for i in location]  # Scale back up face locations since we scaled them down
            color = (255, 0, 0)  # Blue color for known faces
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name} ({match_percentage:.2f}%)", (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Check if 'a' key is pressed to add a new face
    if cv2.waitKey(1) & 0xFF == ord('a'):
        # Prompt user to enter name
        name = input("Enter name of the person: ")

        # Add new face from webcam
        add_new_face_from_webcam(name)

    # Check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
