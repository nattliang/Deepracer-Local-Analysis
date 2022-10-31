import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
import logging
logging.getLogger().setLevel(logging.CRITICAL)


# Find the Number of Episodes Per Iteration
def find_episodes(metrics_list):
  for d in range(len(metrics_list[0])):
    if metrics_list[0][d]['phase'] == 'evaluation':
      trial = int(metrics_list[0][d]['trial'])
      return trial
      
# Parse the Data 
def parse_data(all_metrics, episodes_per_iteration):

    training_data = []
    evaluation_data = []

    for i, metric_file in enumerate(all_metrics):

        current_iteration = 1
        eval_trial = 1
        for episode_data in metric_file:
            if episode_data['phase'] == 'training':
                iteration = current_iteration
                episode = episode_data['episode']
                reward = episode_data['reward_score']
                progress = episode_data['completion_percentage']
                elapsed_time = episode_data['elapsed_time_in_milliseconds']
                trial = episode_data['trial']
                status = episode_data['episode_status']
                complete_time = None
                if status == 'Lap complete':
                    complete_time = elapsed_time

                if trial % episodes_per_iteration == 0:
                    eval_trial = 1
                    current_iteration += 1

                training_data.append([iteration, episode, reward, progress, elapsed_time, complete_time, trial, status])

            if episode_data['phase'] == 'evaluation':
                iteration = current_iteration - 1
                episode = episode_data['episode']
                reward = episode_data['reward_score']
                progress = episode_data['completion_percentage']
                elapsed_time = episode_data['elapsed_time_in_milliseconds']
                trial = eval_trial
                status = episode_data['episode_status']
                complete_time = None
                if status == 'Lap complete':
                    complete_time = elapsed_time

                eval_trial += 1

                evaluation_data.append([iteration, episode, reward, progress, elapsed_time, complete_time, trial, status])

    return [training_data, evaluation_data]

# Convert Episode Data Into Iteration Data
def episode_to_iteration(dataframe):

    sim_df_iterations = dataframe[dataframe["trial"] == 1][["iteration"]].copy()
    sim_df_iterations["avg_progress"] = dataframe.groupby("iteration")["progress"].transform("mean")
    sim_df_iterations["avg_reward"] = dataframe.groupby("iteration")["reward"].transform("mean")
    try:
        sim_df_iterations["avg_elapsed_time"] = dataframe.groupby("iteration")["complete_time"].transform("mean")
        sim_df_iterations["min_elapsed_time"] = dataframe.groupby("iteration")["complete_time"].transform("min")
    except:
        pass

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
            plt.axvline(x=xmax_progress,linestyle='dotted',linewidth=1.5,color='black')
            plt.axhline(y=ymax_progress,linestyle='dotted',linewidth=1.5,color='black')
            plt.gca().text(xmax_progress*0.995, ymax_progress*1.005, f'Max Train Progress @ {xmax_progress}', ha='right', va='bottom', size=font_size)
            
        max_eval_iter = df_slice_e['avg_progress'].idxmax()
        if (max_progress_iter >= 0):
            xmax_progress_e = df_slice_e['iteration'][max_eval_iter]
            ymax_progress_e = df_slice_e['avg_progress'].max()
            plt.axvline(x=xmax_progress_e,linestyle='dotted',linewidth=1.5,color='black')
            plt.axhline(y=ymax_progress_e,linestyle='dotted',linewidth=1.5,color='black')
            plt.gca().text(xmax_progress_e*0.995, ymax_progress_e*1.005, f'Max Training Evaluation @ {xmax_progress_e}', ha='right', va='bottom', size=font_size)
            
        plt.grid(axis='y')
        plt.yticks(np.arange(0, 105, step=10))
        
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        df_slice_iterations.plot(kind='line',linestyle='solid',x='iteration',y='avg_reward',label='Average Training Reward',linewidth=3,color='green',fontsize=font_size,ax=ax2)

        ax2.legend().remove()
        ax2.set_ylabel('Reward', fontsize=font_size)

        plt.plot([], [], ' ', label=f'Iterations: {df_slice_iterations["iteration"].max()}')

        fig.legend(loc="lower center", borderaxespad=0.1, ncol=4, fontsize=14, title="Legend")
        plt.subplots_adjust(bottom=0.15)

        max_rewards_iter = df_slice_iterations['avg_reward'].idxmax()
        if (max_rewards_iter >= 0):
            xmax_rewards = df_slice_iterations['iteration'][max_rewards_iter]
            ymax_rewards = df_slice_iterations['avg_reward'].max()
            plt.axvline(x=xmax_rewards,linestyle='dotted',linewidth=1.5,color='black')
            plt.axhline(y=ymax_rewards,linestyle='dotted',linewidth=1.5,color='black')
            plt.gca().text(xmax_rewards*0.995, ymax_rewards*1.005, f'Max Rewards @ {xmax_rewards}', ha='right', va='bottom', size=font_size)

        plt.show()
        
def plot_pct_competed_laps(df_slice_iterations1):

    font_size=16
    if(len(df_slice_iterations1)>0):
        fig = plt.figure(figsize=(24, 10))
        ax1 = plt.gca()
        
        df_slice_iterations1.plot(kind='line',linestyle=(0,(1,1)),x='iteration',y='pct_completed_laps_SMA',label='Percent Completed Laps SMA',linewidth=3,color='dodgerblue',fontsize=font_size,ax=ax1)
        df_slice_iterations1.plot(kind='line',linestyle='solid',x='iteration',y='avg_progress',label='Average Training Progress',linewidth=3,color='blue',fontsize=font_size,ax=ax1)
        
        ax1.legend().remove()
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=font_size)
        ax1.set_ylabel('Percentage (%)', fontsize=font_size)
        ax1.set_title('Training: Avg. Progress and % Completed Laps', fontsize=20)

        max_progress_iter = df_slice_iterations1['avg_progress'].idxmax()
        if (max_progress_iter >= 0):
            xmax_progress = df_slice_iterations1['iteration'][max_progress_iter]
            ymax_progress = df_slice_iterations1['avg_progress'].max()
            plt.axvline(x=xmax_progress,linestyle='dotted',linewidth=1.5,color='black')
            plt.axhline(y=ymax_progress,linestyle='dotted',linewidth=1.5,color='black')
            plt.gca().text(xmax_progress*0.995, ymax_progress*1.005, f'Max Train Progress @ {xmax_progress}', ha='right', va='bottom', size=font_size)
        
        max_complete_laps_iter = df_slice_iterations1['pct_completed_laps_SMA'].idxmax()
        if (max_complete_laps_iter >= 0):
            xmax_laps = df_slice_iterations1['iteration'][max_complete_laps_iter]
            ymax_laps = df_slice_iterations1['pct_completed_laps_SMA'].max()
            plt.axvline(x=xmax_laps,linestyle='dotted',linewidth=1.5,color='black')
            plt.axhline(y=ymax_laps,linestyle='dotted',linewidth=1.5,color='black')
            plt.gca().text(xmax_laps*0.995, ymax_laps*1.005, f'Max Completed Laps SMA @ {xmax_laps}', ha='right', va='bottom', size=font_size)

        plt.yticks(np.arange(0, 105, step=10))
        
        plt.plot([], [], ' ', label=f'Iterations: {df_slice_iterations1["iteration"].max()}')
        fig.legend(loc="lower center", borderaxespad=0.1, ncol=4, fontsize=14, title="Legend")
        plt.subplots_adjust(bottom=0.15)
        
        plt.yticks(np.arange(0, 105, step=10))
        plt.grid(axis='y')
        plt.show()
    
def plot_episode_end_status(df_slice_iterations):
    font_size=16
    if(len(df_slice_iterations)>0):
        fig = plt.figure(figsize=(24, 10))
        ax1 = plt.gca()
        plt.yticks(np.arange(0, 105, step=10))
        
        df_slice_iterations.plot(kind='line',linestyle=(0,(1,1)),x='iteration',y='pct_completed_laps_SMA',label='Percent Completed Laps SMA',linewidth=2,color='dodgerblue',fontsize=font_size,ax=ax1)
        df_slice_iterations.plot(kind='line',linestyle='dashdot',x='iteration',y='pct_offtrack_SMA',label='Percent Offtrack SMA',linewidth=2,color='brown',fontsize=font_size,ax=ax1)
        df_slice_iterations.plot(kind='line',linestyle='dashdot',x='iteration',y='pct_crashed_SMA',label='Percent Crashed SMA',linewidth=2,color='darkorange',fontsize=font_size,ax=ax1)
                
        ax1.legend().remove()
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=font_size)
        ax1.set_ylabel('Percentage (%)', fontsize=font_size)
        ax1.set_title('Training: Episode End Status', fontsize=20)

        plt.plot([], [], ' ', label=f'Iterations: {df_slice_iterations["iteration"].max()}')
        fig.legend(loc="lower center", borderaxespad=0.1, ncol=4, fontsize=14, title="Legend")
        plt.subplots_adjust(bottom=0.15)
        
        plt.yticks(np.arange(0, 105, step=10))
        plt.grid(axis='y')
        plt.show()

def plot_lap_times(df_slice, df_slice_eval):
    font_size=16
    if df_slice['complete_laps'].sum() > 0:
        df_slice_iterations1 = df_slice[df_slice["complete_laps"] > 0]
        fig = plt.figure(figsize=(24, 10))
        ax1 = plt.gca()
        df_slice_iterations1.plot(kind='line',linestyle='solid',x='iteration',y='avg_elapsed_time',label='Average Elapsed Time',linewidth=2,alpha=0.5,fontsize=font_size,ax=ax1)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
        ax1.set_ylabel(ax1.get_ylabel(), fontsize=20)
        ax1.set_title('Avg Elapsed Time', fontsize=20)
        df_slice_iterations1.plot(kind='line',linestyle='solid',x='iteration',y='min_elapsed_time',label='Minimum Elapsed Time',linewidth=1,color='darkorange',fontsize=font_size,ax=ax1)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
        ax1.set_ylabel('Time (milliseconds)', fontsize=font_size)
        ax1.set_title('Training: Completed Lap Times Per Iteration', fontsize=20)
        plt.grid(axis='y')
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
        plt.grid(axis='y')
        plt.show
    else:
        print("No completed laps during evalution")

def run_notebook(run_env='run.env'):
    # Get the Location of the Metrics Files
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

    print(f'Looking for logs in: data/minio/{BUCKET}/{MODEL_PREFIX}/metrics/')

    metrics_fname = f'data/minio/{BUCKET}/{MODEL_PREFIX}/metrics//TrainingMetrics'
    all_metrics = []

    # Look for up to 10 robomaker workers
    with open(f"{metrics_fname}.json") as metrics_json:
        all_metrics.append(json.load(metrics_json)['metrics'])
    for i in range(1, 10):
        try:
            with open(f"{metrics_fname}_{i}.json") as metrics_json:
                all_metrics.append(json.load(metrics_json)['metrics'])
        except FileNotFoundError:
            break

    # Find the Number of Episodes Per Iteration for Each Robomaker Container
    episodes_per_iteration = find_episodes(all_metrics)
    print(f'# of Episodes per Robomaker: {episodes_per_iteration}')

    # Parse the Data
    training_data, evaluation_data = parse_data(all_metrics, episodes_per_iteration)

    # Convert to Dataframe
    header = ['iteration', 'episode', 'reward', 'progress', 'elapsed_time', 'complete_time', 'trial', 'status']
    sim_df = pd.DataFrame(training_data, columns=header)
    sim_df_E = pd.DataFrame(evaluation_data, columns=header)

    # Convert Episode Data into Iteration Data
    sim_df_iterations = episode_to_iteration(sim_df)
    sim_df_iterations_E = episode_to_iteration(sim_df_E)

    # Plot Graphs
    plot_training_graph(sim_df_iterations, sim_df_iterations_E)
    plot_pct_competed_laps(sim_df_iterations)
    plot_episode_end_status(sim_df_iterations)
    plot_lap_times(sim_df_iterations, sim_df_iterations_E)

