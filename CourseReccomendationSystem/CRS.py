from asyncio.windows_events import NULL
import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def reccomend(x):
    df = pd.read_csv("C:/Users/shaur/OneDrive/Desktop/CourseReccomendationSystem/data/courses.csv")
    df.head()
    df['course_title']
    dir(nfx)
    df['clean_course_title'] = df['course_title'].apply(nfx.remove_stopwords)
    df['clean_course_title'] = df['clean_course_title'].apply(nfx.remove_special_characters)
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(df['clean_course_title'])
    cosine_sim_mat = cosine_similarity(cv_mat)
    course_indices = pd.Series(df.index,index=df['clean_course_title']).drop_duplicates()
    idx = course_indices[x]
    scores = list(enumerate(cosine_sim_mat[idx]))
    sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
    selected_course_indices = [i[0] for i in sorted_scores[1:]]
    selected_course_scores = [i[1] for i in sorted_scores[1:]]
    recommended_result = df['clean_course_title'].iloc[selected_course_indices]
    rec_df = pd.DataFrame(recommended_result)
    rec_df['similarity_scores'] = selected_course_scores

    frame=rec_df.head(20)
    index=[]
    for i in range(20):
        index.append(str(selected_course_indices[i]))
    i=0
    for j in frame['similarity_scores']:
        if j>0:
            i=i+1
    if i==0:
        print("No similiar course found")
    else:     
        print(frame.head(i))           
        plt.bar(index[0:i],frame['similarity_scores'].head(i),width=0.4,color='green')
        plt.xlabel("Course Indices")
        plt.ylabel("Similarity Score")
        plt.title("Similarity Graph")
        plt.show()
    


print("What do you want to study?")
a=input()
df = pd.read_csv("C:/Users/shaur/OneDrive/Desktop/CourseReccomendationSystem/data/courses.csv")
l=list(df['course_title'])
s=len(l)
df.loc[s,'course_title']=a
df.to_csv("C:/Users/shaur/OneDrive/Desktop/CourseReccomendationSystem/data/courses.csv",index=False)
reccomend(a)
df = pd.read_csv("C:/Users/shaur/OneDrive/Desktop/CourseReccomendationSystem/data/courses.csv")
l=list(df['course_title'])
s=len(l)
df=df.drop(s-1)
df.to_csv("C:/Users/shaur/OneDrive/Desktop/CourseReccomendationSystem/data/courses.csv",index=False)

