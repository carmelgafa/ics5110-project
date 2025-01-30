'''This module provides a class for writing a report to an Excel file.'''

import pandas as pd

class ReportWriter():
    '''
    A class for writing a report to an Excel file.
    '''

    def __init__(self, filename:str) -> None:
        '''
        Initialize the ReportWriter with a filename for the report.

        Parameters:
            filename (str): The name of the file where the report will be saved.
        '''
        self.report_filename = filename
        self.workbook_data = {}

    def add_data_frame(self, df, sheet_name:str):
        '''
        Add a Pandas DataFrame to the report.

        Parameters:
            df (Pandas DataFrame): The DataFrame to be added to the report.
            sheet_name (str): The name of the Excel sheet where the DataFrame
                will be written.
        '''
        # Convert to DataFrame if it's not already a DataFrame
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise ValueError(f"Unable to convert input to DataFrame: {e}")

        self.workbook_data[sheet_name] = ('dataframe', df)
        print(f'Added DataFrame sheet: {sheet_name}')

    def add_current_plt(self, img_file, sheet_name):
        '''
        Add a matplotlib plot to the report.

        Parameters:
            img_file (str): The path and filename where the matplotlib figure
                will be saved as a PNG image.
            sheet_name (str): The name of the Excel sheet where the image
                will be inserted.
        '''
        self.workbook_data[sheet_name] = ('img', img_file)
        print(f'Added image sheet: {sheet_name}')

    def save(self):
        '''
        Save the report to an Excel file.

        This method writes the report data and matplotlib images to an Excel file
        specified when the ReportWriter was initialized. The report data is
        written to the specified sheet names, and the matplotlib images are
        inserted into the specified sheet names.
        '''
        with pd.ExcelWriter(
            self.report_filename,
            engine='xlsxwriter'
        ) as writer:
            for sheet_name, payload in self.workbook_data.items():
                if payload[0] == 'dataframe':
                    payload[1].to_excel(writer, sheet_name=sheet_name)
                elif payload[0] == 'img':
                    workbook  = writer.book
                    worksheet = workbook.add_worksheet(sheet_name)
                    worksheet.insert_image(0, 0, payload[1])

        print(f'Saved report to {self.report_filename}')

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a DataFrame
    df_test = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    # Save a plot
    test_img_file = 'plot.png'
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig(test_img_file)
    plt.close()

    # Write report
    rep_writer = ReportWriter('report.xlsx')
    rep_writer.add_data_frame(df_test, 'DataFrameSheet')
    rep_writer.add_current_plt(test_img_file, 'ImageSheet')
    rep_writer.save()
