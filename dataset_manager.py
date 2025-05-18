# IMPORTS
import os
import pandas as pd
from typing import List

# CONSTANTS
FORMATTED_DATA_DIR = "Formatted_Datasets"
RECOLA_USER_INFO = "RECOLA/recola_user_info.xls"
TIME_ADDITION = 2
RECOLA_TIME_STEPS = 1

# CLASS OBJECTS
class Dataset_Manager():
    def __init__(self, reload_recola: bool=False, reload_again: bool=False):

        # LOAD RECOLA DATASET
        self.recola_file_path = f"{FORMATTED_DATA_DIR}/RECOLA.csv"
        self.recola_dataset = None

        if os.path.exists(self.recola_file_path) is True and reload_recola is False:
            self.recola_dataset = pd.read_csv(self.recola_file_path)
        else:
            self.recola_dataset = self.create_recola()
        
        self.temproary_recola = self.recola_dataset

        # LOAD AGAIN DATASET        
        self.again_dataset = None

    # ------------------------------ RECOLA ------------------------------
    
    # FUNCTION TO CREATE THE RECOLA DATASET
    def create_recola(self, output_file_name: str = f"{FORMATTED_DATA_DIR}/RECOLA.csv"):
        
        # STEP 1: GET LIST OF DATAFRAMES FILES
        dataset_dir = "RECOLA/Dataset_RECOLA"
        participant_frames = self.get_dataframes(input_dir=dataset_dir)

        # STEP 2: NORMALISE DATAFRAMES
        participant_frames = self.normalise_dataframes(participants=participant_frames)

        # STEP 3: COMBINE ANNOTATOR COLUMNS
        participant_frames = self.combine_annotators(participants=participant_frames, annotator_suffix="_x", new_header="Annotator_Arousal")
        participant_frames = self.combine_annotators(participants=participant_frames, annotator_suffix="_y", new_header="Annotator_Valence")

        # STEP 4: CREATE WINDOWED DATAFRAMES
        participant_frames = self.create_windowed_dataframes(participants=participant_frames)

        # STEP 5: REMOVE UNNEEDED COLUMNS
        participant_frames = self.remove_columns(participants=participant_frames)
        
        # STEP 6: COMBINE DATAFRAMES
        combined_dataframe = self.combine_dataframes(participants=participant_frames)

        # STEP 7: CREATE CLASS LABELS
        finalised_dataframe = self.create_class_labels(dataframe=combined_dataframe)
        
        # STEP 8: SAVE TO FILE
        finalised_dataframe.to_csv(output_file_name, index=False)

        return finalised_dataframe

    def get_dataframes(self, input_dir: str):
        dataframe_list = [] 

        for file in os.listdir(input_dir):
            number = file.replace(".csv","")
            dataframe = pd.read_csv(f"{input_dir}/{file}")
            dataframe_list.append((number, dataframe))

        return dataframe_list

    def normalise_dataframes(self, participants: List,):
        columns_to_normalise =  list(participants[0][1].filter(regex=f'^{"ComPar"}|{"audio_speech"}|{"VIDEO"}|{"Face_detection"}|{"ECG"}|{"EDA"}|{"FF"}|{"FM"}', axis=1).columns)
        minRange = -1; maxRange = 1
        updated_participants = []
        
        for participant in participants:
            dataframe = participant[1]

            for columnName in columns_to_normalise:
                minValue = dataframe[columnName].min(); maxValue = dataframe[columnName].max()
                dataframe[columnName] = ((dataframe[columnName] - minValue) / (maxValue - minValue)) * (maxRange - minRange) + minRange
            
            updated_participants.append((participant[0], dataframe))
        
        return updated_participants
        
    def combine_annotators(self, participants: List, annotator_suffix: str, new_header: str,):
        annotator_columns = [col for col in participants[0][1].columns if col.endswith(annotator_suffix)]
        updated_participants = []

        for participant in participants:
            dataframe = participant[1]

            annotator_values = []
            for i in range(len(dataframe)):
                sum = 0
                for annotator in annotator_columns: 
                    sum += dataframe[annotator][i]

                average = sum / len(annotator_columns)
                annotator_values.append(average)
            
            dataframe = pd.concat([dataframe, pd.Series(annotator_values, name=new_header)], axis=1)
            updated_participants.append((participant[0], dataframe))
       
        return updated_participants
        
    def create_windowed_dataframes(self, participants: List,):
        # print("Starting CREATE-WINDOWED-DATAFRAMES process")

        updated_participants = []

        for participant in participants:
            dataframe = participant[1]

            start_time = dataframe["time in seconds"][0]
            end_time = start_time + TIME_ADDITION
            time_step = RECOLA_TIME_STEPS
            last_time = dataframe["time in seconds"][len(dataframe)-1]


            windowed_dataframe = None
            while end_time < last_time:
                new_frame = self.obtain_window_frame(window_start=start_time, window_end=end_time, dataframe=dataframe)
                
                if windowed_dataframe is None:
                    windowed_dataframe = new_frame
                else:
                    windowed_dataframe = pd.concat([windowed_dataframe, new_frame], ignore_index=True)

                start_time += time_step
                end_time += time_step

            updated_participants.append((participant[0], windowed_dataframe))
            # print(f"Windowed Dataframe for {participant[0]} completed")

        # print(f"Process CREATE-WINDOWED-DATAFRAMES successfully completed\n")
        return updated_participants

    def create_windowed_dataframes(self, participants: List,):
        # print("Starting CREATE-WINDOWED-DATAFRAMES process")

        updated_participants = []

        for participant in participants:
            dataframe = participant[1]

            start_time = dataframe["time in seconds"][0]
            end_time = start_time + TIME_ADDITION
            time_step = RECOLA_TIME_STEPS
            last_time = dataframe["time in seconds"][len(dataframe)-1]


            windowed_dataframe = None
            while end_time < last_time:
                new_frame = self.obtain_window_frame(window_start=start_time, window_end=end_time, dataframe=dataframe)
                
                if windowed_dataframe is None:
                    windowed_dataframe = new_frame
                else:
                    windowed_dataframe = pd.concat([windowed_dataframe, new_frame], ignore_index=True)

                start_time += time_step
                end_time += time_step

            updated_participants.append((participant[0], windowed_dataframe))
            # print(f"Windowed Dataframe for {participant[0]} completed")

        # print(f"Process CREATE-WINDOWED-DATAFRAMES successfully completed\n")
        return updated_participants

    def obtain_window_frame(self, window_start: float, window_end: float, dataframe: pd.DataFrame):        
        position_list = []

        for i in range(len(dataframe)):
            if dataframe.loc[i, "time in seconds"] >= window_start and dataframe.loc[i, "time in seconds"] < window_end:
                position_list.append(i)
            elif dataframe.loc[i, "time in seconds"] >= window_end:
                break

        if not position_list:  # Check if empty
            return pd.DataFrame()  # Return an empty DataFrame instead of failing
            
        window_dataframe = dataframe.iloc[position_list]
        slice_dictionary = {}
        for column in window_dataframe:
                
                first_row_value = window_dataframe.iloc[0][column]
                if isinstance(first_row_value,(int, float)) and not pd.isna(first_row_value):
                    window_average = window_dataframe[column].mean()
                    slice_dictionary[column] = [window_average]
                else:
                    slice_dictionary[column] = [first_row_value]
        
        return pd.DataFrame(slice_dictionary)

    def remove_columns(self, participants: List):
        # print("Starting REMOVE-COLUMNS process")
        updated_participants = []

        for participant in participants:
            dataframe = participant[1]

            # REMOVE SPECIFIC COLUMNS
            dataframe = dataframe.drop(columns=["Unnamed: 0", "time in seconds"])
            
            for column in dataframe.columns.tolist():
                first_row_value = dataframe.iloc[0][column]
                # REMOVE COLUMNS THAT HAVE NON NUMERICAL ENTRIES
                if isinstance(first_row_value, str) or pd.isna(first_row_value):
                    dataframe = dataframe.drop(columns=[column])
                
                # REMOVE OLD ANNOTATOR FEATURES
                elif column[-2:] == "_x" or column[-2:] == "_y":
                    dataframe = dataframe.drop(columns=[column])
            
            dataframe = dataframe.dropna()
            updated_participants.append((participant[0], dataframe))
            # print(f"Removed columns from dataframe for {participant[0]}")
        
        # print(f"Process REMOVE-COLUMNS successfully completed\n")
        return  updated_participants

    def combine_dataframes(self, participants: List):
        # print("Starting COMBINE-DATAFRAMES process")
        combined_dataframe = None

        for participant in participants:
            participant_number = participant[0]
            dataframe = participant[1]
            dataframe.insert(0, "Participant_Number", participant_number)

            if combined_dataframe is None:
                combined_dataframe = dataframe
            else:
                combined_dataframe = pd.concat([combined_dataframe, dataframe], axis=0, ignore_index=True)
            # print(f"Dataframe for {participant[0]} combined")

        # print(f"Process COMBINE-DATAFRAMES successfully completed\n")
        return combined_dataframe
        
    def create_class_labels(self, dataframe: pd.DataFrame):
        # print("Starting CREATE-CLASS-LABELS process")

        # AROUSAL CLASS LABEL
        class_label, values = "Class_Label_Arousal", []
        median_arousal = dataframe["Annotator_Arousal"].median()

        for index, row in dataframe.iterrows():
            if row["Annotator_Arousal"] > median_arousal:
                values.append(1)
            else:
                values.append(0)

        dataframe.insert(len(dataframe.columns)-1, class_label, values)

        # VALENCE CLASS LABEL
        class_label, values = "Class_Label_Valence", []
        median_arousal = dataframe["Annotator_Valence"].median()

        for index, row in dataframe.iterrows():
            if row["Annotator_Valence"] > median_arousal:
                values.append(1)
            else:
                values.append(0)

        dataframe.insert(len(dataframe.columns), class_label, values)
        # print("Process CREATE-CLASS-LABELS successfully completed")

        return dataframe
    

    # FUNCTION TO OBTAIN ONLY A NUMBER OF PARTICIPANTS
    def get_number_of_participants_frame(self, number_of_participants: int):
        dataset = self.temproary_recola
        
        temp_list = list(dataset["Participant_Number"])
        participant_list = []
        for participant in temp_list:
            if participant not in participant_list:
                participant_list.append(participant)
        
        limited_participant_list = participant_list[:number_of_participants]
        filtered_df = dataset[dataset['Participant_Number'].isin(limited_participant_list)].copy()

        return filtered_df

    # FUNCTION TO REMOVE MODALITIES FROM THE DATASET
    def remove_modality(self, modalities: List):
        parser = {
            "Audio": ["ComPar", "audio_speech"],
            "Video": ["VIDEO", "Face_detection"],
            "Physiology": ["ECG", "EDA"]}
        
        list_of_keywords = []
        for modality in modalities:
            list_of_keywords.append(parser[modality])

        list_of_keywords = [item for sublist in list_of_keywords for item in sublist]

        removed_columns_frame = self.temproary_recola[[col for col in self.temproary_recola.columns if not any(sub in col for sub in list_of_keywords)]].copy()
        return removed_columns_frame

    # FUNCTION TO REMOVE A SPECIFIC CLASS LABEL
    def remove_class_label(self, label_to_keep: str):
        if label_to_keep == "Arousal":
            return self.temproary_recola[[col for col in self.temproary_recola.columns if "Valence" not in col]].copy()
        else:
            return self.temproary_recola[[col for col in self.temproary_recola.columns if "Arousal" not in col]].copy()
        
    def split_by_gender(self, gender: str):
        participant_info = pd.read_excel(RECOLA_USER_INFO)
        participant_info = participant_info.dropna()

        key = "0"
        if gender == "Male":
            key = "1"

        filtered_participants = list(participant_info[participant_info['Sex'].astype(str) == key]["User"])
        filtered_participants = [f"P{str(int(num))}" for num in filtered_participants]

        return self.temproary_recola[self.temproary_recola['Participant_Number'].isin(filtered_participants)].copy()

    # FUNCTION TO COMBINE DATASETS, WITH EACH AS A SEPERATE ENVIRONMENT
    def simplify_environments(self, datasets: List):
        counter = 0
        environment_dataframes = []
        for dataset in datasets:
            dataset["Participant_Number"] = f"P{counter}"
            environment_dataframes.append(dataset)
            counter += 1
        
        combined_df = pd.concat(environment_dataframes, axis=0, ignore_index=True)
        return combined_df








    # FUNCTION TO SAVE ALTERED DATASET
    def save_custom_dataframe(self, file_name: str):
        save_location = f"{FORMATTED_DATA_DIR}/{file_name}"
        self.temproary_recola.to_csv(save_location, index=False)
        temp_dataframe = self.temproary_recola
        self.temproary_recola = self.recola_dataset
        return temp_dataframe

    # ------------------------------ AGAIN -------------------------------
