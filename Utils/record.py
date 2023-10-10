# Author: Tammie li
# Description: result record
# FilePath: \DRL\Utils\record.py

import os
import openpyxl
import shutil

def update_result(score, dataset_name, model_name, subject_id, type):
    result_excel = openpyxl.load_workbook(os.path.join(os.getcwd(), 'StatisResult', f'{dataset_name}', f'Result_{dataset_name}.xlsx'))
    sheet = result_excel[f'{model_name}']

    if True:
        sheet['B'+ str(subject_id+1)] = score.ACC
        sheet['C'+ str(subject_id+1)] = score.BA
        sheet['D'+ str(subject_id+1)] = score.TPR
        sheet['E'+ str(subject_id+1)] = score.FPR
        sheet['F'+ str(subject_id+1)] = score.P
        sheet['G'+ str(subject_id+1)] = score.F1
        sheet['H'+ str(subject_id+1)] = score.KAPPA
        sheet['I'+ str(subject_id+1)] = score.AUC

        # 保存文件
        result_excel.save(os.path.join(os.getcwd(), 'StatisResult', f'{dataset_name}', f'Result_{dataset_name}.xlsx'))

        field = 'pickle' if type is False else 'pth'
        file_path = os.path.join(os.getcwd(), 'Checkpoint', f'{dataset_name}_{model_name}_{subject_id:>02d}.{field}')
        save_path = os.path.join(os.getcwd(), 'SaveParams', f'{dataset_name}')
        shutil.copy(file_path, save_path)

        # 预测结果
        file_path = os.path.join(os.getcwd(), 'PredictionResult', f'{dataset_name}_{model_name}_S{subject_id:>02d}_preds.npy')
        shutil.copy(file_path, save_path)

