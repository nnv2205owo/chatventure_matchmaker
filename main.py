import json
import random

import pandas
import requests
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.stats import halfnorm
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm
from sklearn.naive_bayes import ComplementNB
from random import sample
from PIL import Image
import pickle

pd.set_option('display.max_colwidth', 500)
pd.options.mode.chained_assignment = None  # default='warn'

# Probability dictionary
p = {}

# Movie Genres
movies = ['Adventure',
          'Action',
          'Drama',
          'Comedy',
          'Thriller',
          'Horror',
          'RomCom',
          'Musical',
          'Documentary']

p['Movies'] = [0.28,
               0.21,
               0.16,
               0.14,
               0.09,
               0.06,
               0.04,
               0.01,
               0.01]

# TV Genres
tv = ['Comedy',
      'Drama',
      'Action/Adventure',
      'Suspense/Thriller',
      'Documentaries',
      'Crime/Mystery',
      'News',
      'SciFi',
      'History']

p['TV'] = [0.30,
           0.23,
           0.12,
           0.12,
           0.09,
           0.08,
           0.03,
           0.02,
           0.01]

# Religions (could potentially create a spectrum)
religion = ['Atheist',
            'Catholic',
            'Christian',
            'Jewish',
            'Muslim',
            'Hindu',
            'Buddhist',
            'Spiritual',
            'Agnostic',
            'Other']

p['Religion'] = [0.06,
                 0.16,
                 0.16,
                 0.01,
                 0.19,
                 0.11,
                 0.05,
                 0.10,
                 0.07,
                 0.09]

# Music
music = ['Rock',
         'HipHop',
         'Pop',
         'Country',
         'Latin',
         'EDM',
         'Gospel',
         'Jazz',
         'Classical']

p['Music'] = [0.30,
              0.23,
              0.20,
              0.10,
              0.06,
              0.04,
              0.03,
              0.02,
              0.02]

# Sports
sports = ['Football',
          'Baseball',
          'Basketball',
          'Hockey',
          'Soccer',
          'Other']

p['Sports'] = [0.34,
               0.30,
               0.16,
               0.13,
               0.04,
               0.03]

# Politics (could also put on a spectrum)
politics = ['Liberal',
            'Progressive',
            'Centrist',
            'Moderate',
            'Conservative']

p['Politics'] = [0.26,
                 0.11,
                 0.11,
                 0.15,
                 0.37]

# Social Media
social = ['Facebook',
          'Youtube',
          'Twitter',
          'Reddit',
          'Instagram',
          'Pinterest',
          'LinkedIn',
          'SnapChat',
          'TikTok']

p['Social Media'] = [0.36,
                     0.27,
                     0.11,
                     0.09,
                     0.05,
                     0.03,
                     0.03,
                     0.03,
                     0.03]

# Age (generating random numbers based on half normal distribution)

# Lists of Names and the list of the lists
categories = [movies, religion, music, politics, social, sports]

names = ['Movies', 'Religion', 'Music', 'Politics', 'Social Media', 'Sports']

combined = dict(zip(names, categories))


def string_convert(x):
    """
    First converts the lists in the DF into strings
    """
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x


# Establishing random values for each category


import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pickle

# Use a service account
if not firebase_admin._apps:
    cred = credentials.Certificate('lqdchatventure-firebase-adminsdk-p777u-b92ccc8457.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

query_params = st.experimental_get_query_params()
# print(query_params)
try:
    senderMaskId = query_params['id'][0]
    roomId = query_params['room_id'][0]
except:
    st.error("Đừng có nghịch chứ")
    exit()

with open('meeting_info.pkl', 'rb') as meeting_info_file:
    try:
        meeting_info = pickle.load(meeting_info_file)
    except EOFError:
        meeting_info = {}

with open('id_link.pkl', 'rb') as id_link_file:
    try:
        id_link = pickle.load(id_link_file)
    except EOFError:
        id_link = {}

with open('sender_id_link.pkl', 'rb') as sender_id_link_file:
    try:
        sender_id_link = pickle.load(sender_id_link_file)
    except EOFError:
        sender_id_link = {}

st.title("AI-MatchMaker")

st.header("Finding a Date with Artificial Intelligence")
st.write("Using Machine Learning to Find the Top Dating Profiles for you")

image = Image.open('robot_matchmaker.jpg')

st.image(image, use_column_width=True)

# Instantiating a new DF row to classify later
# profiles_df.drop(profiles_df.columns[len(profiles_df.columns) - 1], axis=1, inplace=True)
if roomId not in id_link:
    # print('Setup room')
    id_link[roomId] = {}
    sender_id_link[roomId] = {}
    # print(id_link, sender_id_link)
    new_profile = pd.DataFrame(columns=['Bios'] + names, index=[0])
elif senderMaskId in id_link[roomId]:
    senderIdLink = id_link[roomId][senderMaskId]
    # print(senderIdLink)
    new_profile = pd.DataFrame(columns=['Bios'] + names, index=[senderIdLink])
else:
    new_profile = pd.DataFrame(columns=['Bios'] + names, index=[id_link[roomId][list(id_link[roomId])[-1]] + 1])

# print("Profiles df : ", profiles_df)
# print("New profile : ", new_profile)

# Asking for new profile data
new_profile['Bios'] = st.text_input("Enter a Bio for yourself: ")

# Printing out some example bios for the user

# Manually inputting the data
for i in new_profile.columns[1:]:
    if i in ['Religion', 'Politics']:
        new_profile[i] = st.selectbox(f"Enter your choice for {i}:", combined[i])

    else:
        options = st.multiselect(f"What is your preferred choice for {i}? ", combined[i])

        # Assigning the list to a specific row
        new_profile.at[new_profile.index[0], i] = options

        new_profile[i] = new_profile[i].apply(lambda x: list(set(x)))

# Looping through the columns and applying the string_convert() function (for vectorization purposes)
# for col in new_profile.columns:
#     new_profile[col] = new_profile[col].apply(string_convert)

# Displaying the User's Profile
st.write("-" * 1000)
st.write("Your profile:")
# print(new_profile)
st.table(new_profile)
# print(new_profile)

# Push to start the matchmaking process
button = st.button("Đăng ký tham gia phòng")
out_button = st.button("Thoát khỏi phòng này")

if out_button:
    sender_mask_ref = db.collection('global_vars').document('masks').collection('users').document(senderMaskId)

    sender_mask_doc = sender_mask_ref.get()
    if not sender_mask_doc.exists:
        print(u'Bạn không tồn tại?')
        st.error('Bạn không tồn tại?')
        exit(1)

    senderId = sender_mask_doc.to_dict()['id']

    room_ref = db.collection('meeting_rooms').document(roomId)
    sender_ref = db.collection('users').document(senderId)
    sender_doc = sender_ref.get()

    room_doc = room_ref.get()
    if not room_doc.exists:
        print(u'Không tồn tại phòng này?')
        st.error('Phòng này không còn tồn tại')
        exit(1)

    room_data = room_doc.to_dict()
    sender_data = sender_doc.to_dict()

    if senderMaskId not in id_link[roomId]:
        st.error('Bạn chưa tham gia phòng này')
    else:

        meeting_info[roomId] = meeting_info[roomId].drop(
            id_link[roomId][senderMaskId])

        del sender_id_link[roomId][id_link[roomId][senderMaskId]]
        del id_link[roomId][senderMaskId]

        room_data['crr_participants'] -= 1

        db.collection('meeting_rooms').document(roomId).update({
            'crr_participants': room_data['crr_participants']
        })

        sender_ref.update({
            'crr_meeting_room': None
        })

        st.success('Bạn đã thoát phòng thành công')

if button:
    sender_mask_ref = db.collection('global_vars').document('masks').collection('users').document(senderMaskId)

    sender_mask_doc = sender_mask_ref.get()
    if not sender_mask_doc.exists:
        print(u'Bạn không tồn tại?')
        st.error('Bạn không tồn tại?')
        exit(1)

    senderId = sender_mask_doc.to_dict()['id']

    room_ref = db.collection('meeting_rooms').document(roomId)
    sender_ref = db.collection('users').document(senderId)
    sender_doc = sender_ref.get()

    room_doc = room_ref.get()
    if not room_doc.exists:
        print(u'Không tồn tại phòng này?')
        st.error('Phòng này không còn tồn tại')
        exit(1)

    room_data = room_doc.to_dict()
    sender_data = sender_doc.to_dict()

    if 'crr_meeting_room' in sender_data and sender_data['crr_meeting_room'] is not None:
        if sender_data['crr_meeting_room'] == roomId:
            st.success('Bạn đã chỉnh sửa hồ sơ thành công')

        else:
            st.success('Bạn đã hủy phòng đăng ký và tham gia phòng này thành công')

            # print(meeting_info[sender_data['crr_meeting_room']])

            meeting_info[sender_data['crr_meeting_room']] = meeting_info[sender_data['crr_meeting_room']].drop(
                id_link[sender_data['crr_meeting_room']][senderMaskId])

            del sender_id_link[roomId][id_link[sender_data['crr_meeting_room']][senderMaskId]]
            del id_link[sender_data['crr_meeting_room']][senderMaskId]

            other_room_data = db.collection('meeting_rooms').document(
                sender_data['crr_meeting_room']).get().to_dict()
            other_room_data['crr_participants'] -= 1

            db.collection('meeting_rooms').document(sender_data['crr_meeting_room']).update({
                'crr_participants': other_room_data['crr_participants']
            })

            room_data['crr_participants'] += 1
    else:
        st.success("Bạn đã tham gia phòng thành công")
        room_data['crr_participants'] += 1

    st.balloons()

    # print(roomId, id_link)
    if roomId not in id_link:
        # print("Setup room")
        id_link[roomId] = {}
        sender_id_link[roomId] = {}

    if roomId not in meeting_info:
        meeting_info[roomId] = pd.DataFrame()

    if senderMaskId in id_link[roomId]:
        senderIdLink = id_link[roomId][senderMaskId]
        print(meeting_info[roomId].iloc[senderIdLink])
        meeting_info[roomId].iloc[senderIdLink] = new_profile.iloc[0]
    else:
        # print(id_link[roomId])
        if len(id_link[roomId]) == 0:
            id_link[roomId][senderMaskId] = 0
            sender_id_link[roomId][0] = senderMaskId
        else:
            print(sender_id_link)
            id_link[roomId][senderMaskId] = id_link[roomId][list(id_link[roomId])[-1]] + 1
            sender_id_link[roomId][id_link[roomId][senderMaskId]] = senderMaskId

        meeting_info[roomId] = pandas.concat([meeting_info[roomId], new_profile])

    if room_data['crr_participants'] == room_data['participants_number']:
        print('Phòng đã đạt đủ người, bắt đầu kết nối sau 15 phút trễ vài giây')

        # print(meeting_info)
        # print(meeting_info[roomId].drop(senderId))
        # print(new_profile)
        df = meeting_info[roomId]

        df['Religion'] = pd.Categorical(df.Religion, ordered=True,
                                        categories=['Catholic',
                                                    'Christian',
                                                    'Jewish',
                                                    'Muslim',
                                                    'Hindu',
                                                    'Buddhist',
                                                    'Spiritual',
                                                    'Other',
                                                    'Agnostic',
                                                    'Atheist'])

        df['Politics'] = pd.Categorical(df.Politics, ordered=True,
                                        categories=['Liberal',
                                                    'Progressive',
                                                    'Centrist',
                                                    'Moderate',
                                                    'Conservative'])

        # Looping through the columns and applying the function
        for col in df.columns:
            df[col] = df[col].apply(string_convert)

        profiles_df = df


        # print(profiles_df)
        # print(new_profile)

        # with open("refined_profiles.pkl", 'wb') as fp:
        #     pickle.dump(df, fp)

        # Modeling the Refined Data
        # Using Clustering then Classification Model

        ## Clustering the Refined Data

        # print(df)

        def vectorization(df, columns):
            """
            Using recursion, iterate through the df until all the categories have been vectorized
            """
            column_name = columns[0]

            # Checking if the column name has been removed already
            if column_name not in ['Bios', 'Movies', 'Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
                return df

            if column_name in ['Religion', 'Politics']:
                df[column_name.lower()] = df[column_name].cat.codes

                df = df.drop(column_name, axis=1)

                return vectorization(df, df.columns)

            else:
                # Instantiating the Vectorizer
                vectorizer = CountVectorizer()

                # Fitting the vectorizer to the Bios
                # print(column_name, df[column_name])
                # print(df[column_name])
                x = vectorizer.fit_transform(df[column_name].values.astype('U'))

                # Creating a new DF that contains the vectorized words
                df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

                # Concating the words DF with the original DF
                new_df = pd.concat([df, df_wrds], axis=1)

                # Dropping the column because it is no longer needed in place of vectorization
                new_df = new_df.drop(column_name, axis=1)

                return vectorization(new_df, new_df.columns)


        # Creating the vectorized DF
        # print(df)
        vect_df = vectorization(df, df.columns)
        # print("First : ")
        # print(vect_df)
        vect_df = vect_df.fillna(0)

        # Scaling
        scaler = MinMaxScaler()

        vect_df = pd.DataFrame(scaler.fit_transform(vect_df), index=vect_df.index, columns=vect_df.columns)
        # print("Second : ")
        # print(vect_df)

        ### PCA

        # Instantiating PCA
        pca = PCA()

        # Fitting and Transforming the DF
        df_pca = pca.fit_transform(vect_df)
        # print("Third : ")
        # print(vect_df)

        # Finding the exact number of features that explain at least 99% of the variance in the dataset
        total_explained_variance = pca.explained_variance_ratio_.cumsum()
        n_over_9 = len(total_explained_variance[total_explained_variance >= .90])
        n_to_reach_9 = vect_df.shape[1] - n_over_9

        # print("PCA reduces the # of features from", vect_df.shape[1], 'to', n_to_reach_9)

        # Reducing the dataset to the number of features determined before

        if n_to_reach_9 > vect_df.shape[0]:
            n_to_reach_9 = vect_df.shape[0]

        pca = PCA(n_components=n_to_reach_9)

        # Fitting and transforming the dataset to the stated number of features
        df_pca = pca.fit_transform(vect_df)

        # Seeing the variance ratio that still remains after the dataset has been reduced
        # print(pca.explained_variance_ratio_.cumsum()[-1])

        ### Performing Hierarchical Agglomerative Clustering
        # - First finding the optimum number of clusters

        # Setting the amount of clusters to test out
        cluster_cnt = [i for i in range(2, 11, 1)]

        # Establishing empty lists to store the scores for the evaluation metrics
        ch_scores = []

        s_scores = []

        db_scores = []

        # The DF for evaluation
        eval_df = df_pca


        ### Helper Function to Evaluate the Clusters

        def cluster_eval(y, x):
            """
            Prints the scores of a set evaluation metric. Prints out the max and min values of the evaluation scores.
            """

            # Creating a DataFrame for returning the max and min scores for each cluster
            df = pd.DataFrame(columns=['Cluster Score'], index=[i for i in range(2, len(y) + 2)])
            df['Cluster Score'] = y

            # print('Max Value:\nCluster #', df[df['Cluster Score'] == df['Cluster Score'].max()])
            # print('\nMin Value:\nCluster #', df[df['Cluster Score'] == df['Cluster Score'].min()])

            # Plotting out the scores based on cluster count
            plt.figure(figsize=(16, 6))
            plt.style.use('bmh')
            plt.plot(x, y)
            plt.xlabel('# of Clusters')
            plt.ylabel('Score')
            plt.show()


        ### Evaluation of Clusters

        # print("The Calinski-Harabasz Score (find max score):")
        # cluster_eval(ch_scores, cluster_cnt)
        #
        # print("\nThe Silhouette Coefficient Score (find max score):")
        # cluster_eval(s_scores, cluster_cnt)
        #
        # print("\nThe Davies-Bouldin Score (find minimum score):")
        # cluster_eval(db_scores, cluster_cnt)

        ### Running HAC
        # Again but with the optimum cluster count

        # Instantiating HAC based on the optimum number of clusters found
        hac = AgglomerativeClustering(n_clusters=3, linkage='complete')

        # Fitting
        hac.fit(df_pca)

        # Getting cluster assignments
        cluster_assignments = hac.labels_

        # Assigning the clusters to each profile
        # df['Cluster #'] = cluster_assignments

        vect_df['Cluster #'] = cluster_assignments

        #### Exporting the Clustered DF and Vectorized DF

        # with open("refined_cluster.pkl", 'wb') as fp:
        #     pickle.dump(df, fp)

        cluster_df = df

        # with open("vectorized_refined.pkl", 'wb') as fp:
        #     pickle.dump(vect_df, fp)

        ## Classification of the New Profile

        ### Importing the Different Classification Models

        # Assigning the split variables
        X = vect_df.drop(["Cluster #"], axis=1)
        y = vect_df['Cluster #']

        # Train, test, split
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        ### Finding the Best Model
        # - Dummy(Baseline Model)
        # - KNN
        # - SVM
        # - NaiveBayes
        # - Logistic Regression
        # - Adaboost

        # NaiveBayes
        nb = ComplementNB()

        # List of models
        models = [nb]

        # List of model names
        names = ['NaiveBayes']

        # Zipping the lists
        classifiers = dict(zip(names, models))

        # Visualization of the different cluster counts
        # vect_df['Cluster #'].value_counts().plot(kind='pie', title='Count of Class Distribution')

        # Since we are dealing with an imbalanced dataset _(because each cluster is not guaranteed to have the same amount of
        # profiles)_, we will resort to using the __Macro Avg__ and __F1 Score__ for evaluating the performances of each model.

        # Dictionary containing the model names and their scores
        models_f1 = {}

        # Looping through each model's predictions and getting their classification reports
        for name, model in tqdm(classifiers.items()):
            # Fitting the model
            model.fit(X_train, y_train)

            # print('\n' + name + ' (Macro Avg - F1 Score):')

            # Classification Report
            report = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=True)
            f1 = report['macro avg']['f1-score']

            # Assigning to the Dictionary
            models_f1[name] = f1

            # print(f1)

        # Fitting the Best Model to our Dataset
        # _(Optional: Tune the model with GridSearch)_

        # Fitting the model
        nb.fit(X, y)

        # Saving the Classification Model for future use

        # dump(nb, "refined_model.joblib")

        model = nb


        # print(vect_df)

        def new_vectorization(df, columns, input_df):
            """
            Using recursion, iterate through the df until all the categories have been vectorized
            """

            column_name = columns[0]

            # Checking if the column name has been removed already
            if column_name not in ['Bios', 'Movies', 'Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
                return df, input_df

            # Encoding columns with respective values
            if column_name in ['Religion', 'Politics']:

                # Getting labels for the original df
                df[column_name.lower()] = df[column_name].cat.codes

                # Dictionary for the codes
                d = dict(enumerate(df[column_name].cat.categories))

                d = {v: k for k, v in d.items()}

                # Getting labels for the input_df
                input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]

                # Dropping the column names
                input_df = input_df.drop(column_name, axis=1)

                df = df.drop(column_name, axis=1)

                return new_vectorization(df, df.columns, input_df)

            # Vectorizing the other columns
            else:
                # Instantiating the Vectorizer
                vectorizer = CountVectorizer()

                # Fitting the vectorizer to the columns
                x = vectorizer.fit_transform(df[column_name].values.astype('U'))

                y = vectorizer.transform(input_df[column_name])

                # Creating a new DF that contains the vectorized words
                df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

                y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names_out(), index=input_df.index)

                # Concating the words DF with the original DF
                new_df = pd.concat([df, df_wrds], axis=1)

                y_df = pd.concat([input_df, y_wrds], axis=1)

                # Dropping the column because it is no longer needed in place of vectorization
                new_df = new_df.drop(column_name, axis=1)

                y_df = y_df.drop(column_name, axis=1)

                return new_vectorization(new_df, new_df.columns, y_df)


        def top_ten(cluster, vect_df, input_vect):
            """
            Returns the DataFrame containing the top 10 similar profiles to the new data
            """
            # print("Top ten")
            # print(vect_df)
            # Filtering out the clustered DF
            des_cluster = vect_df[vect_df['Cluster #'] == cluster[0]].drop('Cluster #', axis=1)
            # print(des_cluster)

            # Appending the new profile data
            des_cluster = pd.concat([input_vect, des_cluster], sort=False)

            # Finding the Top 10 similar or correlated users to the new user
            user_n = input_vect.index[0]

            # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
            corr = des_cluster.T.corrwith(des_cluster.loc[user_n])
            # print(corr)

            # Creating a DF with the Top 10 most similar profiles
            top_10_sim = corr.sort_values(ascending=False)
            # print("Nguyên Vũ desu")
            # print(top_10_sim)
            # print("Ohayo")

            # The Top Profiles
            # top_10 = profiles_df.loc[top_10_sim.index]

            return top_10_sim.astype('object')


        def scaling(df, input_df):
            """
            Scales the new data with the scaler fitted from the previous data
            """
            scaler = MinMaxScaler()

            scaler.fit(df)

            input_vect = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)

            return input_vect


        def example_bios():
            """
            Creates a list of random example bios from the original dataset
            """
            # Example Bios for the user
            st.write("-" * 100)
            st.text("Some example Bios:\n(Try to follow the same format)")
            for i in sample(list(profiles_df.index), 3):
                st.text(profiles_df['Bios'].loc[i])
            st.write("-" * 100)


        ## Interactive Section

        # Creating the Titles and Image
        with st.spinner('Finding your Top 10 Matches...'):
            paired_users_check = []
            paired_users = []
            remain_list = []
            for i in profiles_df.index:
                if i in paired_users_check:
                    continue
                if i not in sender_id_link[roomId]:
                    continue
                new_profile = profiles_df.loc[[i]]
                # Vectorizing the New Data
                # print(new_profile)
                # for col in df.columns:
                #     profiles_df[col] = profiles_df[col].apply(string_convert)
                #     new_profile[col] = new_profile[col].apply(string_convert)

                print(profiles_df)
                print(new_profile)
                df_v, input_df = new_vectorization(profiles_df, profiles_df.columns, new_profile)

                # Scaling the New Data
                new_df = scaling(df_v, input_df)

                # Predicting/Classifying the new data
                cluster = model.predict(new_df)

                # Finding the top 10 related profiles
                top_10_df = top_ten(cluster, vect_df, new_df)

                # Success message

                # Displaying the Top 10 similar profiles

                # st.table(top_10_df)
                print(id_link[roomId])
                # print(top_10_df)
                print("Hey : ", i)
                for j in top_10_df.index:
                    if j == i:
                        continue
                    if j in paired_users_check:
                        continue
                    if j not in sender_id_link[roomId]:
                        continue

                    if isinstance(j, int):
                        paired_users_check.append(i)
                        paired_users_check.append(j)

                        paired_users.append([sender_id_link[roomId][j], sender_id_link[roomId][i]])
                        break

                print()
                print("End hey", i)
                # st.table(top_10_df)
                # print(profiles_df)
                # for i in profiles_df.index:
                #     crr_profile = profiles_df.loc[[i]]
                #     print(crr_profile)

            for i in profiles_df.index:
                if i not in paired_users_check:
                    remain_list.append(sender_id_link[roomId][i])

            random.shuffle(remain_list)
            r = requests.post("https://lqdchatventure-web.herokuapp.com/meeting_rooms",
                              data={
                                  'action': 'start',
                                  'data': json.dumps({'action': 'start', 'room_id': roomId, 'remain_list': remain_list,
                                                      'paired_users': paired_users})
                              })

            room_ref.delete()

            st.success("Found your Top 10 Most Similar Profiles!")
            st.balloons()
            st.write(paired_users)
            st.write(remain_list)


    else:
        room_ref.update({
            'crr_participants': room_data['crr_participants']
        })

        sender_ref.update({
            'crr_meeting_room': roomId
        })

        with open('meeting_info.pkl', 'wb') as meeting_info_file:
            pickle.dump(meeting_info, meeting_info_file)

        with open('id_link.pkl', 'wb') as id_link_file:
            pickle.dump(id_link, id_link_file)

        with open('sender_id_link.pkl', 'wb') as sender_id_link_file:
            pickle.dump(sender_id_link, sender_id_link_file)

    print(meeting_info)
    # print(meeting_info[roomId].drop(senderId))
    print(new_profile)
