import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
import logging
logging.getLogger().setLevel(logging.CRITICAL)

def json_to_list(file):
    file = file.replace("{","[")
    file = file.replace("}","]")
    file = file.replace('"', '')
    file = file.replace('[metrics: ',"")
    file = file.replace(', version: 2.0, best_model_metric: progress]',"")
    file = file.replace('reward_score: ', '"')
    file = file.replace(', metric_time: ', '","')
    file = file.replace(', start_time: ', '","')
    file = file.replace(', elapsed_time_in_milliseconds: ', '","')
    file = file.replace(', episode: ', '","')
    file = file.replace(', trial: ', '","')
    file = file.replace(', phase: ', '","')
    file = file.replace(', completion_percentage: ', '","')
    file = file.replace(', episode_status: ', '","')
    file = file.replace(']', '"]')
    file = file.replace('"]"]', '"]]')
    #print(file)
    return json.loads(file)

def find_episodes(data_list):
  for d in range(len(data_list)):
    if data_list[d][6] == 'evaluation':
      trial = int(data_list[d][5])
      return trial
      
def parse_data(data_list, episodes_per_iteration):
    training = []
    evaluation = []
    x = 1
    y = 1
    for d in range(len(data_list)):
        if data_list[d][6] == 'training':
            iteration = x
            episode = int(data_list[d][4])
            reward = float(data_list[d][0])
            progress = float(data_list[d][7])
            elapsed_time = float(data_list[d][3])
            trial = int(data_list[d][5])
            trial = trial % episodes_per_iteration
            status = data_list[d][8]
            if status == 'Lap complete':
                complete_time = elapsed_time
            else:
                complete_time = None
            if trial == 0:
                trial = episodes_per_iteration
                x += 1
            training.append([iteration, episode, reward, progress, elapsed_time, complete_time, trial, status])
        elif data_list[d][6] == 'evaluation':
            iteration = x - 1
            episode = int(data_list[d][4])
            reward = float(data_list[d][0])
            progress = float(data_list[d][7])
            elapsed_time = float(data_list[d][3])
            trial = y
            status = data_list[d][8]
            if status == 'Lap complete':
                complete_time = elapsed_time
            else:
                complete_time = None
            y += 1
            try:
                if data_list[d+1][6] == 'training':
                    y = 1
            except:
                pass
            evaluation.append([iteration, episode, reward, progress, elapsed_time, complete_time, trial, status])
            
    return [training, evaluation]

def episode_to_iteration(dataframe):
    sim_df_iterations = dataframe[dataframe["trial"] == 1][["iteration"]].copy()
    sim_df_iterations["avg_progress"] = dataframe.groupby("iteration")["progress"].transform("mean")
    sim_df_iterations["avg_reward"] = dataframe.groupby("iteration")["reward"].transform("mean")
    sim_df_iterations["avg_elapsed_time"] = dataframe.groupby("iteration")["complete_time"].transform("mean")
    sim_df_iterations["min_elapsed_time"] = dataframe.groupby("iteration")["complete_time"].transform("min")

    episode_count = dataframe.groupby("iteration")["episode"].count()
    sim_df_iterations = pd.merge(sim_df_iterations,episode_count,on="iteration",how="outer")
    sim_df_iterations.rename(columns={"episode":"episodes"}, inplace=True)
    
    complete_laps = dataframe[dataframe["status"] == "Lap complete"].groupby("iteration")["status"].count()
    sim_df_iterations = pd.merge(sim_df_iterations, complete_laps,on="iteration",how="outer")
    sim_df_iterations.rename(columns={"status":"complete_laps"}, inplace=True)
    sim_df_iterations["complete_laps"].fillna(0, inplace=True)
    sim_df_iterations["complete_laps"] = sim_df_iterations["complete_laps"].astype(int)
    sim_df_iterations["pct_completed_laps"] = (sim_df_iterations["complete_laps"] / sim_df_iterations["episodes"] * 100.0)
    
    offtrack_count = dataframe[dataframe["status"] == "Off track"].groupby("iteration")["status"].count()
    sim_df_iterations = pd.merge(sim_df_iterations, offtrack_count,on="iteration",how="outer")
    sim_df_iterations.rename(columns={"status":"offtrack_count"}, inplace=True)
    sim_df_iterations["offtrack_count"].fillna(0, inplace=True)
    sim_df_iterations["offtrack_count"] = sim_df_iterations["offtrack_count"].astype(int)
    sim_df_iterations["pct_offtrack"] = (sim_df_iterations["offtrack_count"] / sim_df_iterations["episodes"] * 100.0)
    
    crash_count = dataframe[dataframe["status"] == "Crashed"].groupby("iteration")["status"].count()
    sim_df_iterations = pd.merge(sim_df_iterations, crash_count,on="iteration",how="outer")
    sim_df_iterations.rename(columns={"status":"crash_count"}, inplace=True)
    sim_df_iterations["crash_count"].fillna(0, inplace=True)
    sim_df_iterations["crash_count"] = sim_df_iterations["crash_count"].astype(int)
    sim_df_iterations["pct_crashed"] = (sim_df_iterations["crash_count"] / sim_df_iterations["episodes"] * 100.0)
    
    # Create a simple moving average for completed laps, offtrack count, and crash count to reduce deviation
    try:
        sim_df_iterations['pct_completed_laps_SMA'] = sim_df_iterations['pct_completed_laps'].rolling(window=3).mean()
        sim_df_iterations['pct_offtrack_SMA'] = sim_df_iterations['pct_offtrack'].rolling(window=3).mean()
        sim_df_iterations['pct_crashed_SMA'] = sim_df_iterations['pct_crashed'].rolling(window=3).mean()
    except:
        pass

    return sim_df_iterations

def plot_training_graph(df_slice_iterations, df_slice_e):
    font_size=16
    if(len(df_slice_iterations)>0):
        fig = plt.figure(figsize=(24, 10))
        ax = plt.gca()  # gca stands for 'get current axis'
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # force integer labels on x-axis
        
        df_slice_iterations.plot(kind='line',linestyle='solid',x='iteration',y='avg_progress',label='Average Training Progress',linewidth=3,color='blue',fontsize=font_size,ax=ax)
        df_slice_e.plot(kind='line',linestyle='solid',x='iteration',y='avg_progress',label='Average Evaluation Progress',linewidth=3,color='red',fontsize=font_size,ax=ax)
        ax.legend().remove()
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel('Progress', fontsize=font_size)
        ax.set_title('Training & Evaluation Graph', fontsize=20)

        max_progress_iter = df_slice_iterations['avg_progress'].idxmax()
        if (max_progress_iter >= 0):
            xmax_progress = df_slice_iterations['iteration'][max_progress_iter]
            ymax_progress = df_slice_iterations['avg_progress'].max()
            plt.axvline(x=xmax_progress,linestyle='dotted',linewidth=0.75,color='black')
            plt.axhline(y=ymax_progress,linestyle='dotted',linewidth=0.75,color='black',alpha=0.3)
            plt.gca().text(xmax_progress*0.995, ymax_progress*1.005, 'Max Train Progress @ %d' % xmax_progress, ha='right', va='bottom', size=font_size)
            
        max_eval_iter = df_slice_e['avg_progress'].idxmax()
        if (max_progress_iter >= 0):
            xmax_progress_e = df_slice_e['iteration'][max_eval_iter]
            ymax_progress_e = df_slice_e['avg_progress'].max()
            plt.axvline(x=xmax_progress_e,linestyle='dotted',linewidth=0.75,color='black')
            plt.axhline(y=ymax_progress_e,linestyle='dotted',linewidth=0.75,color='black',alpha=0.3)
            plt.gca().text(xmax_progress_e*0.995, ymax_progress_e*1.005, 'Max Training Evaluation @ %d' % xmax_progress_e, ha='right', va='bottom', size=font_size)
            
        plt.yticks(np.arange(0, 105, step=10))
        
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        df_slice_iterations.plot(kind='line',linestyle='solid',x='iteration',y='avg_reward',label='Average Training Reward',linewidth=3,color='green',fontsize=font_size,ax=ax2)

        ax2.legend().remove()
        ax2.set_ylabel('Reward', fontsize=font_size)

        plt.plot([], [], ' ', label='Iterations: %d' % df_slice_iterations["iteration"].max())

        fig.legend(loc="lower center", borderaxespad=0.1, ncol=4, fontsize=14, title="Legend")
        plt.subplots_adjust(bottom=0.15)

        max_rewards_iter = df_slice_iterations['avg_reward'].idxmax()
        if (max_rewards_iter >= 0):
            xmax_rewards = df_slice_iterations['iteration'][max_rewards_iter]
            ymax_rewards = df_slice_iterations['avg_reward'].max()
            plt.axvline(x=xmax_rewards,linestyle='dotted',linewidth=0.75,color='black')
            plt.axhline(y=ymax_rewards,linestyle='dotted',linewidth=0.75,color='black',alpha=0.3)
            plt.gca().text(xmax_rewards*0.995, ymax_rewards*1.005, 'Max Rewards @ %d' % xmax_rewards, ha='right', va='bottom', size=font_size)

        plt.show()
        
def plot_pct_competed_laps(df_slice_iterations1, df_slice_iterations2):
    font_size=16
    if(len(df_slice_iterations1)>0):
        fig = plt.figure(figsize=(24, 10))
        ax1 = fig.add_subplot(121)
        
        df_slice_iterations1.plot(kind='line',linestyle=(0,(1,1)),x='iteration',y='pct_completed_laps_SMA',label='Percent Completed Laps SMA',linewidth=3,color='dodgerblue',fontsize=font_size,ax=ax1)
        df_slice_iterations1.plot(kind='line',linestyle='solid',x='iteration',y='avg_progress',label='Average Training Progress',linewidth=3,color='blue',fontsize=font_size,ax=ax1)
        
        ax1.legend().remove()
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=font_size)
        ax1.set_ylabel('Percentage (%)', fontsize=font_size)
        ax1.set_title('Robomaker 1: Avg. Progress and % Completed Laps', fontsize=20)

        max_progress_iter = df_slice_iterations1['avg_progress'].idxmax()
        if (max_progress_iter >= 0):
            xmax_progress = df_slice_iterations1['iteration'][max_progress_iter]
            ymax_progress = df_slice_iterations1['avg_progress'].max()
            plt.axvline(x=xmax_progress,linestyle='dotted',linewidth=0.75,color='black')
            plt.axhline(y=ymax_progress,linestyle='dotted',linewidth=0.75,color='black',alpha=0.3)
            plt.gca().text(xmax_progress*0.995, ymax_progress*1.005, 'Max Train Progress @ %d' % xmax_progress, ha='right', va='bottom', size=font_size)
        
        max_complete_laps_iter = df_slice_iterations1['pct_completed_laps_SMA'].idxmax()
        if (max_complete_laps_iter >= 0):
            xmax_laps = df_slice_iterations1['iteration'][max_complete_laps_iter]
            ymax_laps = df_slice_iterations1['pct_completed_laps_SMA'].max()
            plt.axvline(x=xmax_laps,linestyle='dotted',linewidth=0.75,color='black')
            plt.axhline(y=ymax_laps,linestyle='dotted',linewidth=0.75,color='black',alpha=0.3)
            plt.gca().text(xmax_laps*0.995, ymax_laps*1.005, 'Max Completed Laps SMA @ %d' % xmax_laps, ha='right', va='bottom', size=font_size)

        plt.yticks(np.arange(0, 105, step=10))
        
        ax2 = fig.add_subplot(122)
        df_slice_iterations2.plot(kind='line',linestyle=(0,(1,1)),x='iteration',y='pct_completed_laps_SMA',label='',linewidth=3,color='dodgerblue',fontsize=font_size,ax=ax2)
        df_slice_iterations2.plot(kind='line',linestyle='solid',x='iteration',y='avg_progress',label='',linewidth=3,color='blue',fontsize=font_size,ax=ax2)
        ax2.legend().remove()
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=font_size)
        ax2.set_ylabel('Percentage (%)', fontsize=font_size)
        ax2.set_title('Robomaker 2: Avg. Progress and % Completed Laps', fontsize=20)

        max_progress_iter = df_slice_iterations2['avg_progress'].idxmax()
        if (max_progress_iter >= 0):
            xmax_progress = df_slice_iterations2['iteration'][max_progress_iter]
            ymax_progress = df_slice_iterations2['avg_progress'].max()
            plt.axvline(x=xmax_progress,linestyle='dotted',linewidth=0.75,color='black')
            plt.axhline(y=ymax_progress,linestyle='dotted',linewidth=0.75,color='black',alpha=0.3)
            plt.gca().text(xmax_progress*0.995, ymax_progress*1.005, 'Max Train Progress @ %d' % xmax_progress, ha='right', va='bottom', size=font_size)
        
        max_complete_laps_iter = df_slice_iterations2['pct_completed_laps_SMA'].idxmax()
        if (max_complete_laps_iter >= 0):
            xmax_laps = df_slice_iterations2['iteration'][max_complete_laps_iter]
            ymax_laps = df_slice_iterations2['pct_completed_laps_SMA'].max()
            plt.axvline(x=xmax_laps,linestyle='dotted',linewidth=0.75,color='black')
            plt.axhline(y=ymax_laps,linestyle='dotted',linewidth=0.75,color='black',alpha=0.3)
            plt.gca().text(xmax_laps*0.995, ymax_laps*1.005, 'Max Completed Laps SMA @ %d' % xmax_laps, ha='right', va='bottom', size=font_size)

        plt.plot([], [], ' ', label='Iterations: %d' % df_slice_iterations1["iteration"].max())
        fig.legend(loc="lower center", borderaxespad=0.1, ncol=4, fontsize=14, title="Legend")
        plt.subplots_adjust(bottom=0.15)
        
        plt.yticks(np.arange(0, 105, step=10))
        plt.show()
    
def plot_episode_end_status(df_slice_iterations1, df_slice_iterations2):
    font_size=16
    if(len(df_slice_iterations1)>0):
        fig = plt.figure(figsize=(24, 10))
        ax1 = fig.add_subplot(121)
        plt.yticks(np.arange(0, 105, step=10))
        
        df_slice_iterations1.plot(kind='line',linestyle=(0,(1,1)),x='iteration',y='pct_completed_laps_SMA',label='Percent Completed Laps SMA',linewidth=2,color='dodgerblue',fontsize=font_size,ax=ax1)
        df_slice_iterations1.plot(kind='line',linestyle='dashdot',x='iteration',y='pct_offtrack_SMA',label='Percent Offtrack SMA',linewidth=2,color='brown',fontsize=font_size,ax=ax1)
        df_slice_iterations1.plot(kind='line',linestyle='dashdot',x='iteration',y='pct_crashed_SMA',label='Percent Crashed SMA',linewidth=2,color='darkorange',fontsize=font_size,ax=ax1)
                
        ax1.legend().remove()
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=font_size)
        ax1.set_ylabel('Percentage (%)', fontsize=font_size)
        ax1.set_title('Robomaker 1: Episode End Status', fontsize=20)
        
        ax2 = fig.add_subplot(122)
        
        df_slice_iterations2.plot(kind='line',linestyle=(0,(1,1)),x='iteration',y='pct_completed_laps_SMA',label='',linewidth=2,color='dodgerblue',fontsize=font_size,ax=ax2)
        df_slice_iterations2.plot(kind='line',linestyle='dashdot',x='iteration',y='pct_offtrack_SMA',label='',linewidth=2,color='brown',fontsize=font_size,ax=ax2)
        df_slice_iterations2.plot(kind='line',linestyle='dashdot',x='iteration',y='pct_crashed_SMA',label='',linewidth=2,color='darkorange',fontsize=font_size,ax=ax2)

        ax2.legend().remove()
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=font_size)
        ax2.set_ylabel('Percentage (%)', fontsize=font_size)
        ax2.set_title('Robomaker 2: Episode End Status', fontsize=20)

        plt.plot([], [], ' ', label='Iterations: %d' % df_slice_iterations1["iteration"].max())
        fig.legend(loc="lower center", borderaxespad=0.1, ncol=4, fontsize=14, title="Legend")
        plt.subplots_adjust(bottom=0.15)
        
        plt.yticks(np.arange(0, 105, step=10))
        plt.show()

def plot_lap_times(df_slice_1, df_slice_2, df_slice_eval):
    font_size=16
    if df_slice_1['complete_laps'].sum() > 0:
        df_slice_iterations1 = df_slice_1[df_slice_1["complete_laps"] > 0]
        fig = plt.figure(figsize=(24, 10))
        ax1 = fig.add_subplot(121)
        df_slice_iterations1.plot(kind='line',linestyle='solid',x='iteration',y='avg_elapsed_time',label='Average Elapsed Time',linewidth=2,alpha=0.5,fontsize=font_size,ax=ax1)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
        ax1.set_ylabel(ax1.get_ylabel(), fontsize=20)
        ax1.set_title('Avg Elapsed Time', fontsize=20)
        df_slice_iterations1.plot(kind='line',linestyle='solid',x='iteration',y='min_elapsed_time',label='Minimum Elapsed Time',linewidth=1,color='darkorange',fontsize=font_size,ax=ax1)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
        ax1.set_ylabel('Time (milliseconds)', fontsize=font_size)
        ax1.set_title('Robomaker #1: Completed Lap Times Per Iteration', fontsize=20)

    if df_slice_2['complete_laps'].sum() > 0:
        df_slice_iterations2 = df_slice_2[df_slice_2["complete_laps"] > 0]
        ax2 = fig.add_subplot(122)
        df_slice_iterations2.plot(kind='line',linestyle='solid',x='iteration',y='avg_elapsed_time',label='Average Elapsed Time',linewidth=2,alpha=0.5,fontsize=font_size,ax=ax2)
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=20)
        ax2.set_ylabel(ax2.get_ylabel(), fontsize=20)
        ax2.set_title('Avg Elapsed Time', fontsize=20)
        df_slice_iterations2.plot(kind='line',linestyle='solid',x='iteration',y='min_elapsed_time',label='Minimum Elapsed Time',linewidth=1,color='darkorange',fontsize=font_size,ax=ax2)
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=20)
        ax2.set_ylabel('Time (milliseconds)', fontsize=font_size)
        ax2.set_title('Robomaker #2: Completed Lap Times Per Iteration', fontsize=20)

    if df_slice_1['complete_laps'].sum() > 0 or df_slice_2['complete_laps'].sum() > 0:
        plt.show
    else:
        print("No completed laps during training")
    
    if df_slice_eval['complete_laps'].sum() > 0:
        df_slice_e = df_slice_eval[df_slice_eval["complete_laps"] > 0]       
        fig = plt.figure(figsize=(24, 10))
        ax1 = plt.gca()
        df_slice_e.plot(kind='line',linestyle='solid',x='iteration',y='avg_elapsed_time',label='Average Elapsed Time',linewidth=2,alpha=0.5,fontsize=font_size,ax=ax1)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
        ax1.set_ylabel(ax1.get_ylabel(), fontsize=20)
        ax1.set_title('Avg Elapsed Time', fontsize=20)
        df_slice_e.plot(kind='line',linestyle='solid',x='iteration',y='min_elapsed_time',label='Minimum Elapsed Time',linewidth=1,color='darkorange',fontsize=font_size,ax=ax1)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
        ax1.set_ylabel('Time (milliseconds)', fontsize=font_size)
        ax1.set_title('Evaluation: Completed Lap Times Per Iteration', fontsize=20)
        plt.show

def run_notebook():
    # Get the Location of the Metrics Files
    run_env = 'run.env'
    system_env = 'system.env'
    with open(run_env, 'r') as run:
        for line in run.readlines():
            if "DR_LOCAL_S3_MODEL_PREFIX=" in line:
                spl_pnt = '='
                MODEL_PREFIX = line.partition(spl_pnt)[2]
                MODEL_PREFIX = MODEL_PREFIX.replace('\n','')
    with open(system_env, 'r') as sys:
        for line in sys.readlines():
            if "DR_LOCAL_S3_BUCKET=" in line:
                spl_pnt = '='
                BUCKET = line.partition(spl_pnt)[2]
                BUCKET = BUCKET.replace('\n','')
    print('Looking for logs in: data/minio/%s/%s/metrics/TrainingMetrics.json' % (BUCKET,MODEL_PREFIX))
    try:
        metrics1_fname = 'data/minio/%s/%s/metrics/TrainingMetrics.json' %(BUCKET,MODEL_PREFIX)
        metrics2_fname = 'data/minio/%s/%s/metrics/TrainingMetrics_1.json' %(BUCKET,MODEL_PREFIX)
    except:
        print('No logs found in data/minio/%s/%s/metrics/' %(BUCKET,MODEL_PREFIX))

    # Compile the Metrics Logs so They Can be Read by this Notebook
    with open(metrics1_fname, 'r') as file1:
        for line in file1:
            metrics1 = json_to_list(line)
    with open(metrics2_fname, 'r') as file2:
        for line in file2:
            metrics2 = json_to_list(line)

    # Find the Number of Episodes Per Iteration for Each Robomaker Container
    episodes_per_iteration = find_episodes(metrics1)
    print('# of Episodes per Robomaker: %s' %episodes_per_iteration)

    # Parse the Data
    training_data1, evaluation_data1 = parse_data(metrics1, episodes_per_iteration)
    training_data2, evaluation_data2 = parse_data(metrics2, episodes_per_iteration)

    # Convert to Dataframe
    header = ['iteration', 'episode', 'reward', 'progress', 'elapsed_time', 'complete_time', 'trial', 'status']
    sim_df1 = pd.DataFrame(training_data1, columns=header)
    sim_df2 = pd.DataFrame(training_data2, columns=header)
    sim_df_E = pd.DataFrame(evaluation_data1, columns=header)

    # Convert Episode Data into Iteration Data
    sim_df_iterations1 = episode_to_iteration(sim_df1)
    sim_df_iterations2 = episode_to_iteration(sim_df2)
    sim_df_iterations_E = episode_to_iteration(sim_df_E)

    # Plot Graphs
    plot_training_graph(sim_df_iterations1, sim_df_iterations_E)
    plot_training_graph(sim_df_iterations2, sim_df_iterations_E)
    plot_pct_competed_laps(sim_df_iterations1, sim_df_iterations2)
    plot_episode_end_status(sim_df_iterations1, sim_df_iterations2)
    plot_lap_times(sim_df_iterations1, sim_df_iterations2, sim_df_iterations_E)



