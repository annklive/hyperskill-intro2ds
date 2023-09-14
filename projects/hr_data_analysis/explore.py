import pandas as pd
import requests
import os

# scroll down to the bottom to implement your solution
if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('A_office_data.xml' not in os.listdir('../Data') and
        'B_office_data.xml' not in os.listdir('../Data') and
        'hr_data.xml' not in os.listdir('../Data')):
        print('A_office_data loading.')
        url = "https://www.dropbox.com/s/jpeknyzx57c4jb2/A_office_data.xml?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/A_office_data.xml', 'wb').write(r.content)
        print('Loaded.')

        print('B_office_data loading.')
        url = "https://www.dropbox.com/s/hea0tbhir64u9t5/B_office_data.xml?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/B_office_data.xml', 'wb').write(r.content)
        print('Loaded.')

        print('hr_data loading.')
        url = "https://www.dropbox.com/s/u6jzqqg1byajy0s/hr_data.xml?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/hr_data.xml', 'wb').write(r.content)
        print('Loaded.')

        # All data in now loaded to the Data folder.

    # write your code here
    data_path = '../Data'
    officeA = pd.read_xml(os.path.join(data_path, 'A_office_data.xml'))
    officeB = pd.read_xml(os.path.join(data_path, 'B_office_data.xml'))
    hr = pd.read_xml(os.path.join(data_path, 'hr_data.xml'))

    officeA.index = "A" + officeA.employee_office_id.astype(str)
    officeB.index = "B" + officeB.employee_office_id.astype(str)
    hr.index = hr.employee_id

    # stage 2/5
    all_offices = pd.concat([officeA, officeB])
    final_data = all_offices.merge(hr, left_index=True, right_index=True, indicator=True, how='left')
    final_data.dropna(inplace=True)
    final_data.drop(['employee_id', 'employee_office_id', '_merge'], axis=1, inplace=True)
    final_data.sort_index(inplace=True)
    #print(merged_data.index.tolist())
    #print(merged_data.columns.tolist())

    # stage 3/5
    hardworking_dept = final_data.sort_values('average_monthly_hours', ascending=False).iloc[:10,:].Department.tolist()
    # %%
    low_salary_IT_projects = final_data[(final_data.Department == 'IT') & (final_data.salary == 'low')].loc[:,
                             'number_project'].sum()
    #print(hardworking_dept)
    #print(low_salary_IT_projects)
    #print(final_data.loc[['A4', 'B7064', 'A3033'], ['last_evaluation', 'satisfaction_level']].values.tolist())

    # stage 4/5
    def count_bigger_5(nprojects_series):
        return (nprojects_series > 5).sum()


    results = final_data.groupby('left').agg({
        'number_project': ['median', count_bigger_5],
        'time_spend_company': ['mean', 'median'],
        'Work_accident': ['mean'],
        'last_evaluation': ['mean', 'std']
    }).round(2)

    # print(results.to_dict())

    # stage 5/5
    first_table = final_data.pivot_table(index='Department', columns=['left', 'salary'], values='average_monthly_hours',
                                         aggfunc='median')
    first_table = first_table.loc[(first_table[0, 'high'].values < first_table[0, 'medium']) &
                                  (first_table[1, 'low'].values < first_table[1, 'high']), :]



    second_table = final_data.pivot_table(index='time_spend_company', columns='promotion_last_5years',
                                          values=['satisfaction_level', 'last_evaluation'],
                                          aggfunc=['min', 'max', 'mean'])
    second_table = second_table.loc[second_table['mean', 'last_evaluation', 0].values > second_table[
        'mean', 'last_evaluation', 1].values, :]

    print(first_table.to_dict())
    print(second_table.to_dict())


