import streamlit as st
from skimage.metrics import structural_similarity
import tempfile 
import os
import cv2
import numpy as np
from PIL import Image

def option1_code():
    # Code for option 1
    st.write("Option 1 selected!")
    # Add your code here for option 1
    from kmeans_for_colourClustering import find_number_of_colors
    from hackrx_TextMeasure import rate_text_amount 
    from keras.models import load_model

    # import pandas as pd # pip install pandas
    # from matplotlib import pyplot as plt # pip install matplotlib
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#Used to avoid printing of Warnings
    #st.title("Deep Learning based Visual Testing Tool for Website Design Validation Developed by PixelProphets")
    st.title("Accessibility Testing Tool")
    # get the path/directory
    folder_dir = "C://aditi//competitions//hackerx4//test"

    # streamlite commands

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #st.title('Website Design Assessment Tool using Deep Learning')
    st.text(
        'Upload the Image from the listed category.\n[good design or bad design]')

    saved_model1 = load_model('C://aditi//competitions//hackerx4//best_model_mobilenet1.hdf5')
    saved_model2 = load_model('C://aditi//competitions//hackerx4//best_model_mobilenet_forContrast.hdf5')
    saved_model3 = load_model('C://aditi//competitions//hackerx4//best_model_mobilenet_forLayout.hdf5')
    saved_model4 = load_model("C://aditi/competitions/hackerx4/best_model_mobilenet_forClutter.hdf5")

    uploaded_file = st.file_uploader("Choose an image...", type=[
                                    "jpg", "png", "tiff", "bmp"])
    if uploaded_file is not None:
        imge = Image.open(uploaded_file)
        st.image(imge, caption='Uploaded Image')

        if st.button('PREDICT'):
            Categories1 = ['bad_website_design', 'good_website_design']
            Categories2 = ['bad_website_contrast', 'good_website_contrast']
            Categories3 = ['bad_website_layout', 'good_website_layout']
            Categories4 = ['cluttered_website', 'non_cluttered_website']

            st.write('Result.....')
            flat_data = []
            imge = np.array(imge)
            print(imge.shape)
            # The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.
            img1 = cv2.resize(imge, (224, 224), interpolation=cv2.INTER_NEAREST)
            norm_image = cv2.normalize(
                img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if norm_image.shape[2] == 4:
                #slice off the alpha channel
                norm_image = norm_image[:, :, :3]

            print(norm_image.shape) 

            y = np.expand_dims(norm_image, axis=0)
            print(y.shape)

            y_out1 = saved_model1.predict(y)
            y_out1 = np.round(y_out1)

            y_out2 = saved_model2.predict(y)
            y_out2 = np.round(y_out2)

            y_out3 = saved_model3.predict(y)
            y_out3 = np.round(y_out3)

            y_out6 = saved_model4.predict(y)
            y_out6 = np.round(y_out6)

            y_out1 = Categories1[y_out1.argmax()]
            y_out2 = Categories2[y_out2.argmax()]
            y_out3 = Categories3[y_out3.argmax()]
            y_out6 = Categories4[y_out6.argmax()]

            rating = 0

            if y_out2=='good_website_contrast':
                rating = rating + 1
            if y_out3=='good_website_layout':
                rating = rating + 1
            if y_out6=='non_cluttered_website':
                rating = rating + 1

            totalUniqueColors = find_number_of_colors(uploaded_file) 
            y_out4="Less than or Equal to 6 colors detected"
            if totalUniqueColors>6:
                y_out4="More than 6 colors detected"

            y_out5 = rate_text_amount(uploaded_file)

            #st.title(f' PREDICTED OUTPUT: {y_out1}')
            st.title(f' PREDICTED OUTPUT: {y_out2}')
            #st.title(f' PREDICTED OUTPUT: {y_out3}')
            st.title(f' PREDICTED OUTPUT: {y_out4}')
            st.title(f' PREDICTED OUTPUT: {y_out5}')
            #st.title(f' PREDICTED OUTPUT: {y_out6}')
            #st.title(f' PREDICTED RATING: {rating}/3')

            # q = saved_model.predict_proba(y)
            # for index, item in enumerate(Categories):
            #   st.write(f'{item} : {q[0][index]*100}%')

    st.text("")
    st.text('Made by PixelProphets')

def option2_code():
    # Code for option 2
    st.write("Option 2 selected!")
    # Streamlit app
    st.title("Visual Testing Tool")

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.text('Upload the Image from the listed category.\n[good design or bad design]')
    sensitivity = st.slider("Select the sensitivity of size filter:", min_value=250, max_value=40000, value=50, step=50)


    uploaded_file1 = st.file_uploader("Choose the first photo", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Choose the second photo", type=["jpg", "jpeg", "png"])

    if uploaded_file1 and uploaded_file2 is not None:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption='First Photo', use_column_width=True)
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption='Second photo')
        
        

        # Save the images to temporary files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file1, \
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file2:
            image1.save(temp_file1.name)
            image2.save(temp_file2.name)

            # Read the images using cv2.imread()
            before = cv2.imread(temp_file1.name)
            after = cv2.imread(temp_file2.name)

            if st.button('COMPARE'):
                st.write('Result.....')
                # Convert images to grayscale
                before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
                after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

                # Compute SSIM between the two images
                (score, diff) = structural_similarity(before_gray, after_gray, full=True)
                st.write("Image Similarity: {:.4f}%".format(score * 100))

                # The diff image contains the actual image differences between the two images
                # and is represented as a floating point data type in the range [0,1] 
                # so we must convert the array to 8-bit unsigned integers in the range
                # [0,255] before we can use it with OpenCV
                diff = (diff * 255).astype("uint8")
                diff_box = cv2.merge([diff, diff, diff])

                # Threshold the difference image, followed by finding contours to
                # obtain the regions of the two input images that differ
                thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                mask = np.zeros(before.shape, dtype='uint8')
                filled_after = after.copy()

                for c in contours:
                    area = cv2.contourArea(c)
                    if area >40000-sensitivity:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
                        cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
                        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
                        cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                        cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

                beforergb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
                afterrgb = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)
                diffrgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
                diff_boxrgb = cv2.cvtColor(diff_box, cv2.COLOR_BGR2RGB)
                maskrgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                filled_afterrgb = cv2.cvtColor(filled_after, cv2.COLOR_BGR2RGB)
                # Display the result image using streamlit
                result_image = Image.fromarray(afterrgb)
                st.image(result_image, caption='Result photo')

                # After displaying the result, remove the temporary files
                #os.unlink(temp_file1.name)
                #os.unlink(temp_file2.name)
    # Add your code here for option 2
    st.text("")
    

# Streamlit app code
def main():

    st.title("Website Testing Tool ")
    st.title("Select an Option")
    
    # Create a selectbox to choose between options
    selected_option = st.selectbox("Choose an option:", ("Accessibility Testing", "Visual Testing"))
    
    # Execute corresponding code based on the selected option
    if selected_option == "Accessibility Testing":
        option1_code()
    elif selected_option == "Visual Testing":
        option2_code()
    

main()