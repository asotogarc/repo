import streamlit as st

def main():
    st.title("Webcam Photo Capture")
    
    # Use st.camera_input to capture an image from the webcam
    img_file_buffer = st.camera_input("Take a picture")
    
    # Check if a photo was taken
    if img_file_buffer is not None:
        # Display the captured image
        st.image(img_file_buffer, caption="Captured Image")
        st.success("Photo captured successfully!")
    else:
        st.info("Waiting for you to take a photo...")

if __name__ == "__main__":
    main()
