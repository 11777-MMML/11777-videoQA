import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
models = ['ATP', 'EIGV', 'HGA', 'IGV', 'VGT']

def generate_all_results_all_wrong_all_correct():
    df_cols = ['qid', 'answer']
    df_cols.extend(models)
    df = pd.DataFrame(columns=df_cols)

    test_next_csv = pd.read_csv('./next_test.csv')
    test_next_csv = test_next_csv.rename(columns={'qid': 'next_qid'})
    test_next_csv['qid'] = test_next_csv['video'].astype(str) + '_' + test_next_csv['next_qid'].astype(str)
    test_next_csv = test_next_csv.drop(['answer', 'next_qid', 'video', 'width', 'height'], axis=1)

    for model in models:
        model_df = pd.read_csv(f'./{model}_test_results.csv')
        model_df = model_df.rename(columns={'prediction': model})

        if df.empty:
            df = model_df
        else:
            model_df = model_df.drop('answer', axis=1)
            df = df.set_index('qid').join(model_df.set_index('qid')).reset_index()

    df = df.set_index('qid').join(test_next_csv.set_index('qid')).reset_index()

    # df['all_correct'] = (df['answer'] == df[models[0]]) & (df['answer'] == df[models[1]]) & (
    #             df['answer'] == df[models[2]]) & (df['answer'] == df[models[3]]) & (df['answer'] == df[models[4]])

    # exclude IGV
    df['all_correct'] = (df['answer'] == df[models[0]]) & (df['answer'] == df[models[1]]) & (
            df['answer'] == df[models[2]]) & (df['answer'] == df[models[4]])

    # df['all_wrong'] = (df['answer'] != df[models[0]]) & (df['answer'] != df[models[1]]) & (
    #             df['answer'] != df[models[2]]) & (df['answer'] != df[models[3]]) & (df['answer'] != df[models[4]])

    # exclude IGV
    df['all_wrong'] = (df['answer'] != df[models[0]]) & (df['answer'] != df[models[1]]) & (
            df['answer'] != df[models[2]]) & (df['answer'] != df[models[4]])

    for model in models:
        df[f'{model}_correct'] = df['answer'] == df[model]

    df.to_csv('all_test_results.csv')
    df[df['all_correct'] == True].to_csv('all_correct_without_IGV.csv')
    df[df['all_wrong'] == True].to_csv('all_wrong_without_IGV.csv')


def plot_frame_counts(df: pd.DataFrame):
    # df = df[df['frame_count']<4000]
    df.reset_index()
    sns.scatterplot(data=df, x=df.index, y='frame_count', hue='type')
    plt.show()


# plot_frame_counts(pd.read_csv('./all_wrong.csv'))
generate_all_results_all_wrong_all_correct()