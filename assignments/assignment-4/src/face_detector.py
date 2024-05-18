import os
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import seaborn as sns


def initialize_MTCNN():
    """
    The function initializes the pretrained Multi-Task Cascaded Convolutional Neural Networks
    (MTCNN) for the face detection task.
    """
    mtcnn = MTCNN(keep_all = True)
    return mtcnn


def initialize_df(filepath):
    """
    The function initializes an empty pandas dataframe with predefined columns to store the
    results from the face detection task. The dataframe will have a column indicating the
    newspaper, the decade to which the newpaper issue (i.e., image) belongs to, the total
    number of faces detected in a given image, whether there are faces in the given image
    (0 = no, 1 = yes) and the percentage of pages that have faces on them.
    """
    df = pd.DataFrame(columns = ("Newspaper", "Decade", "Number of faces",
                                "Present face", "Pages with faces (%)"))
    
    newspaper = filepath.split('-')[0]
    df.loc[len(df)] = [newspaper, 0, 0, 0, 0]
    return df


def get_decade(year):
    """
    The function calculates the decade for a given year by removing the last digit and
    replacing it with a zero. For example year 1789 will be transformed to 1780.
    """
    decade = str(year)[:3] + "0"
    return decade


def face_detection(image, mtcnn, df, newspaper, decade):
    """
    The function detects faces in a given input image using the pretrained MTCNN
    model and updates the dataframe with results.
    """ 
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        detected_faces = boxes.shape[0]
        df.loc[len(df)] = [newspaper, decade, detected_faces, 1, 0]
    else:
        df.loc[len(df)] = [newspaper, decade, 0, 0, 0]
    return df 


def process_newspaper(filepath, mtcnn, df):
    """
    The function iterates through all issues of the three newspapers. Then, it calls the
    function face_detection() to detect faces in the issue and updates the dataframe accordingly. 
    """
    for newspaper in sorted(os.listdir(filepath)):
        newspaper_path = os.path.join(filepath, newspaper)
        for issue in sorted(os.listdir(newspaper_path)):
            issue_year = int(issue.split('-')[1])
            decade = get_decade(issue_year)
            issue_dir = os.path.join(newspaper_path, issue)
            print(issue_dir)
            image = Image.open(issue_dir)
            df = face_detection(image, mtcnn, df, newspaper, decade)
    return df


def calculate_pages_with_faces(df):
    """
    The function calculates the total number of faces, total number of pages, and
    percentage of pages with faces.
    """
    total_faces = df.groupby(["Newspaper", "Decade"]).agg({"Number of faces": "sum", "Present face": "sum"}).reset_index()
    total_pages = df.groupby(["Newspaper", "Decade"]).size().reset_index(name = "Number of pages")
    pages_with_faces = pd.merge(total_faces, total_pages, on = ["Newspaper", "Decade"])
    pages_with_faces["Pages with faces (%)"] = (pages_with_faces["Present face"] / pages_with_faces["Number of pages"]) * 100
    pages_with_faces["Decade"] = pages_with_faces["Decade"].astype(str)
    return pages_with_faces


def plot(pages_with_faces, outpath):
    """
    Plots the percentage of pages with faces per decade and save the plot as an image.
    """
    pages_with_faces = pages_with_faces[pages_with_faces['Newspaper'].isin(['GDL', 'IMP', 'JDG'])]
    sns.relplot(data = pages_with_faces, kind = "line", x = "Decade", y = "Pages with Faces (%)",
                hue = "Newspaper", palette = "Paired")  
    plt.xticks(rotation = 45, fontsize = 8)
    plt.title('Percentage of pages with faces per pecade', fontsize = 12)
    plt.tight_layout()
    plt.savefig(outpath)
    return print("The plot has been saved to the out folder")


def save_df_to_csv(df, csv_outpath):
    """
    Save the dataframe as .csv to a specified outpath.
    """
    df.to_csv(csv_outpath)
    return print("The results have been saved to the out folder")


def main():
    
    filepath = os.path.join("in")

    mtcnn = initialize_MTCNN()

    df = initialize_df(filepath)

    df = process_newspaper(filepath, mtcnn, df)

    pages_with_faces = calculate_pages_with_faces(df)
    save_df_to_csv(pages_with_faces, "out/face_count.csv")

    plot(pages_with_faces, "out/face_plot.png")

if __name__ == "__main__":
    main()