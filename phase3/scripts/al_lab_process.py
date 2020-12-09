import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd
from pprint import pprint

SPREADSHEET_ID = '1e17hvwdR4HtzREdg40es0HGCSUfNcacDQfhVUr0amRQ'
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
RANGE_NAME = 'Round 4!A1:I8'

def connect_gsheet():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    return sheet

def get_al_result_sheet(sheet_name, sheet_range='A2:I8'):
    sheet = connect_gsheet()
    col_name_range = f"{sheet_name}!A1:I1"
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=col_name_range).execute()
    columns = result.get('values', [])

    if sheet_range[:2] == 'A1':
        sheet_range = f'A2{sheet_range[2:]}'
    RANGE = f"{sheet_name}!{sheet_range}"
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE).execute()
    data = result.get('values', [])

    if not data:
        print('No data found.')
    else:
        df = pd.DataFrame(data, columns=columns[0])
        #df.set_index('name')
        return df

def write_result_sheet(sheet_range, values, sheet_name='Test'):
    sheet = connect_gsheet()
    value_input_option = 'USER_ENTERED'
    RANGE = f'{sheet_name}!{sheet_range}'
    value_range_body = {'values': values}
    
    request = sheet.values().update(spreadsheetId=SPREADSHEET_ID, range=RANGE, valueInputOption=value_input_option, body=value_range_body)
    response = request.execute()

    # TODO: Change code below to process the `response` dict:
    pprint(response)



if __name__=='__main__':
    """
    raw_data = [['name', 'Index', 'Vial location', 'Reagent1 (ul) ', 'Reagent2 (ul) ', 'Reagent3 (ul)', 'Reagent6 (ul)', 'Reagent7 (ul)', 'Crystal Score'], ['DT0', '9279', 'A7', '130', '0', '270', '50', '50', '1'], ['DT1', '19082', 'B7', '0', '0', '500', '0', '0', '1'], ['KNN0', '3', 'C7', '4', '30', '186', '140', '140', '3'], ['KNN1', '3', 'D7', '4', '30', '186', '140', '140', '3'], ['MIT', '2880', 'E7', '42', '200', '58', '100', '100', '3'], ['PLT0', '9716', 'F7', '358', '10', '32', '50', '50', '1'], ['PLT1', '2378', 'G7', '6', '280', '4', '105', '105', '4']]
    columns = raw_data[0]
    data = raw_data[1:]
    df = pd.DataFrame(data, columns=columns)
    df = df.set_index('name')
    """
    write_result_sheet()
