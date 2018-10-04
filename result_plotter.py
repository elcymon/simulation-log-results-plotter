import time 
import glob
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics
from copy import deepcopy 
import pathlib #for creating directories
def find_index(the_list):
	for i in range(0,len(the_list)):
		print(i,the_list[i])

def process_readme(filename):
	#reads the data in readm.txt file and saves them in a list
	#this is done by appending experiment parameters to a list
	#thereby forming a list of lists
	#where each list represents an experiment
	#readme = [['par1:val1','par2:val2','par3:val3'...]
	#			['par1:val1','par2:val2','par3:val3'...]
	#			['par1:val1','par2:val2','par3:val3'...]
	#			...]
	
	##id_map holds the mapping between experiments, start times and names (algorithm name).
	#id_map = {id : [name,[time_list]]}
	#t2id_map {t1:id,t2:id,...}
	global readme,id_map,t2id_map
	with open(filename) as file:
		for data in file:
			data = data.rstrip('\n')
			d = data.split(',')
			if len(d) > 1:
				exp_id = d[0]
				exp_id = exp_id.split(':')
				stat = d[-2]
				stat = stat.split(':')
				start_time = stat[1]
				paras = ['nei_sensing','turn_prob_max']
				name = ''
				for x in paras:
					for y in d:
						if x in y:
							name = name + y.replace(':','=') + ','
				name = name.rstrip(',')
				t2id_map[start_time] = exp_id[1]#map time to id 
				if exp_id[1] in id_map:
					
					id_map[exp_id[1]][1].append(start_time)
				else:
					id_map[exp_id[1]] = [name,[start_time]]
				readme.append(d)
			
def get_exp_time(filename,file_category):
	#from the file name, this function extracts the time
	#an experiment began. The time is returned as a string.
	#print(filename)
	st = filename.rfind('\\')
	en = filename.find(file_category)
	return filename[st+1:en]

def process_litter_count(filename):
	#litter_counts = [['exp1_time',[t_max,max_lit],[time_list],[lit_count_list]],
	#					['exp2_time',[t_max,max_lit],[time_list],[lit_count_list]],
	#					['exp3_time',[t_max,max_lit],[time_list],[lit_count_list]],
	#					['exp4_time',[t_max,max_lit],[time_list],[lit_count_list]],
	#					['exp5_time',[t_max,max_lit],[time_list],[lit_count_list]],
	#					...]
	global litter_counts
	exp_name = get_exp_time(filename,'_litter_count')
	t = []
	count = []
	max_value = [-5,-5]
	exp_data = [exp_name,max_value,t,count]
	x = 0
	with open(filename) as file:
		for data in file:
			if x >= valid_record:
				data = data.rstrip('\n')
				data_list = data.split(',')
				data_list = [float(i) for i in data_list] #OR list(map(float,data_list))
				
				if(data_list[1] > max_value[1]):
					max_value[0] = data_list[0]
					max_value[1] = data_list[1]
					#print(data_list)
				t.append(data_list[0])
				count.append(data_list[1])
			x +=1
	litter_counts.append(exp_data)
			
def process_robot_data(filename):
	#Record robot data for all experiments
	#robot_files = {'exp1_time': [['robot1',robot1_t1,robot1_t2,...],
	#								['robot2',robot2_t1,robot2_t2,...],
	#								['robot3',robot3_t1,robot3_t2,...],...],
	#				'exp2_time': [['robot1',robot1_t1,robot1_t2,...],
	#								['robot2',robot2_t1,robot2_t2,...],
	#								['robot3',robot3_t1,robot3_t2,...],...]
	#				...}
	#where robotn = name of n-th robot
	# robotn_tm = logged data of n-th robot at m-th time step
	# robot logged data is currently in format:
	# [t, x, y, yaw, turn_prob, seen_lit, nei_seen, comm_sig, lit_carried, 'status']
	global robot_files
	exp_name = get_exp_time(filename,'_m_4wrobot')
	robot_data = []
	robot_name = ''
	x = 0
	with open(filename) as file:
		for data in file:
			if x >= valid_record:
				data = data.rstrip('\n')
				data = data.split(':')
				robot_name = data[0]
				robot_status = data[2]
				d = data[1].split(',')
				#print(d[:-1])
				#data_to_float = [float(i) for i in d[:-1]]
				#data_to_float.append(d[-1])
				data_to_float = [float(i) for i in d]
				data_to_float.append(robot_status)
				robot_data.append(data_to_float)
			x +=1
	robot_data.insert(0,robot_name)
	if exp_name in robot_files:
		robot_files[exp_name].append(robot_data)
	else:
		robot_files[exp_name] = []
		robot_files[exp_name].append(robot_data)
		
			
def get_experiments_filenames(directory):
	global readme,id_map,t2id_map,litter_counts,robot_files
	
	files = []
	for name in glob.glob(directory + '*'):
		files.append(name)
		#print(name)
		
	files.sort()
	loaded = False
	s = str(files)
	if ('readme.npy' in s and
			'id_map.npy' in s and
			't2id_map.npy' in s and
			'litter_counts.npy' in s and
			'robot_files.npy' in s):
		readme = np.load(directory + 'readme.npy').tolist()
		id_map = np.load(directory + 'id_map.npy').item()
		t2id_map = np.load(directory + 't2id_map.npy').item()
		litter_counts = np.load(directory + 'litter_counts.npy').tolist()
		robot_files = np.load(directory + 'robot_files.npy').item()
		loaded = True
		print('fi')
	else:
		one_experiment = True
		for name in files:
			if "readme.md" in name:
				process_readme(name)
			elif "_litter_count" in name:
				process_litter_count(name)
			elif "_m_4wrobot" in name and one_experiment:
				#print(name)
				one_experiment = True
				process_robot_data(name)
		loaded = True
		np.save(directory + 'readme.npy',readme)
		np.save(directory + 'id_map.npy',id_map)
		np.save(directory + 't2id_map.npy',t2id_map)
		np.save(directory + 'litter_counts.npy',np.array(litter_counts,dtype=object))
		np.save(directory + 'robot_files.npy',robot_files)
		print('else')
		#print(robot_files)
			
	return files
	#print(name)
def plot_parameter_trend(data,p,p_name):
	sf = plt.figure()
	ax = sf.add_subplot(1,1,1)
	robot_name = data[0]
	x_values = [i[0] for i in data[1:]]
	y_values = [i[p] for i in data[1:]]
	#print(x_values)
	#print(y_values)
	ax.set_title(robot_name + ': ' + p_name)
	#ax.axis('square')
	ax.plot(x_values,y_values,label=str(p))
	ax.legend(loc=2,bbox_to_anchor=(1,1))
	
def plot_x_parameter_trend(alg_t,data,p,p_names):
	#plt.figure()
	ff,axes = plt.subplots(len(p),1)
	
	robot_name = data[0]
	
	x = [i[0] for i in data[1:]]
	
	n = 0
	alg_id = t2id_map[alg_t]
	axes[n].set_title(alg_id+":"+robot_name+","+alg_t)
	for par in p:
		y = [i[par] for i in data[1:]]
		
		axes[n].plot(x,y)
		axes[n].set_ylabel(p_names[n])
		n = n + 1
	ff.savefig(plotsNdata+robot_name+'turnProb_repulsionSensed_attractionSensed'+fName+'.pdf')
	
	
	
def robots_trajectory():
	param_name = ['time', 'xloc', 'yloc','orientation','turn_prob',
				'seen_litter', 'seen_rep_neighbours', 'repulsion_sensed',
				'seen_att_neighbours','attraction_sensed',
				'litter_carried','linear_distance','angular_distance']
	all_experiments = list(robot_files.keys())
	all_experiments.sort()
	#print(all_experiments)
	#expmt_choice = ''
	alg_id = '30'
	#print(t2id_map)
	#for xxx in all_experiments:
		#print(t2id_map[xxx])
	#	if t2id_map[xxx] == alg_id:
	#		expmt_choice = xxx
	#		break
	print(len(all_experiments))
	expmt_choice = [all_experiments[0]]#,all_experiments[30],all_experiments[60]]
	
	#plot_x_parameter_trend(expmt_choice,robot_files[expmt_choice][7],[4,7,9],
		#					[param_name[4],param_name[7],param_name[9]])
		
	n_plots = len(all_experiments)
	print(n_plots)
	#f_rob = plt.figure()
	#ax_rob = f_rob.add_subplot(1,2,1)	
	index = 1
	f_rob_list = []
	for expmt in expmt_choice:#[all_experiments[0]]:
		#f_rob,ax_rob = plt.subplots(2,1)
		#f_rob,ax_rob = plt.subplots()
		f_robNax = plt.subplots()
		f_rob,ax_rob = f_robNax
		#f_rob_list.append(f_robNax)
		#ax_rob = f_rob.add_subplot(1,2,1)	
		#ax = f.add_subplot(n_plots/2,2,index)
		xx = t2id_map[expmt]
		expname = 'ID'+xx+','+expmt
		#expname = 'ID:10,'+id_map['10'][0]
		
		ax_rob.set_title(expname)#[0]
		index = index + 1
		x = []
		y = []
		#for p in range(len(param_name)):
			#plot_parameter_trend(robot_files[expmt][1],p,param_name[p])
		for robot in robot_files[expmt]:
			#plot_x_parameter_trend(expmt,robot,[4,7,9],
			#				[param_name[4],param_name[7],param_name[9]])
		#robot = robot_files['2017_12_19_13_56_14'][1]
			x_values = [i[1] for i in robot[1:]]
			y_values = [i[2] for i in robot[1:]]
			x.extend(x_values)
			y.extend(y_values)
		#print(len(robot[2]))
			#ax.set_xlim(xmin=-10,xmax=10)
			#ax.set_ylim([-10,10])
		#ax.legend(loc=2,bbox_to_anchor=(1,1))
		
		
		#x = [int(i) for i in x[::1]]
		#for i in range(10): x.extend(x) 
		#y = [int(i) for i in y[::1]]
		#for i in range(10): y.extend(y) 
		nb = np.linspace(-25,25,4)
		ax_rob.set_xlim(-25,25)#[0]
		ax_rob.set_ylim(-25,25)#[0]
		#ax_rob[0].axis('square')
		
		#plt.axis([-25,25,-25,25])
		#weights = [100] * len(y)
		#ax_rob.axis('equal')
		nbins = (nb,nb)
		H, xedges, yedges = np.histogram2d(x,y,bins=nbins)#,weights=weights)
		H = np.rot90(H)
		H = np.flipud(H)
		#print(H.shape)
		Hmasked = np.ma.masked_where(H==0,H)
		#ax_rob.plot(xedges,yedges,Hmasked)#,label=robot[0])
		#ax.axis('equal')
		im = mpl.image.NonUniformImage(ax_rob,interpolation='bilinear')#[0]
		xcenters = (xedges[:-1] + xedges[1:]) / 2
		ycenters = (yedges[:-1] + yedges[1:]) / 2
		im.set_data(xcenters,ycenters,H)
		pcm = ax_rob.imshow(H,interpolation='bilinear')#[0],vmin = 1000, vmax = 6500
		ax_rob.images.append(im)#[0]
		#ax_rob[0].scatter([0],[0],marker='.',color='b')
		
		#cbar.set_ylabel('Counts')
		#X,Y = np.meshgrid(xedges,yedges)
		#pcm = ax_rob[0].imshow(H,interpolation='bilinear')
		
		#pcm = ax_rob[0].pcolormesh(xedges,yedges,Hmasked)
		#pcm = ax_rob[0].hist2d(x,y,bins=nbins)
		
		divider = make_axes_locatable(ax_rob)#[0]
		cax = divider.append_axes("right",size="5%",pad=0.25)
		#print(cax.get_xlim(),cax.get_ylim())
		f_rob.colorbar(pcm,cax=cax)
		
		#ax_rob2 = f_rob.add_subplot(1,2,2)
		#ax_rob[1].set_xlim(-25,25)
		#ax_rob[1].set_ylim(-25,25)
		#ax_rob[1].axis('square')
		#ax_rob[1].axis([-25,25,-25,25])
		
		#ax_rob[1].scatter(x,y,marker='.',s=1,linewidths=1)
		
		f_rob.tight_layout()
		f_rob.savefig(plotsNdata+'robot_trajectory_'+expname+fName+'.pdf')#, bbox_inches='tight')
	#plt.show()
	
		
	#f.tight_layout(pad=0.0,h_pad=1,w_pad=1)
	
	#f.savefig(folder_name+'robots_pose_data.pdf')
def litter_pcts():			
	global litter_counts
	#pct_tlist = []
	#pct_litlist = []
	#litt_pcts.reverse()
	
	for t in range(len(litter_counts)):
		l = litter_counts[t]
		time = l[1][0]
		litCount = l[1][1]
		
		for t_loc in range(len(l[3])):
			if l[3][t_loc] >=lit_pcts:
				 
				#t_loc = next(x[0] for x in enumerate(l[3]) if x[1] >= pl)
				time = l[2][t_loc]
				litCount = l[3][t_loc]
				break;
		l[1][0] = time
		l[1][1] = litCount
			
			#pct_tlist.append(l[2][t_loc])
			#pct_litlist.append(l[3][t_loc])
		#pct_tlist.append(l[1][0])
		#pct_litlist.append(l[1][1])
		#l.append(pct_tlist)
		#l.append(pct_litlist)
		
def litter_collected():
	#litter_bar_plot = [i[1][1] for i in litter_counts]
	#litter_bar_plot_exps = [i[0] for i in litter_counts]
	mpl.style.use('default')# to see available styles use print(plt.style.available)
	
	litter_pcts()#append a list of time and litter count percents for each experiment
	C = []
	c = 0
	figx = []
	for p in range(len(id_map)):
		x = plt.subplots()
		figx.append(x)
	
	#litter_bar_plot_lit = [[]] * (len(lit_pcts)+1)
	#litter_bar_plot_t = [[]] * (len(lit_pcts)+1)
	litter_bar_plot = []
	litter_bar_plot_t = []
	
	litter_bar_plot_exps = []
	id_list = [] # keeps track of all the experiment id's that has been encountered
	id_num = []
	for t in litter_counts:
		exp_id = t2id_map[t[0]]
		if exp_id in id_list:
			x = id_list.index(exp_id)
			litter_bar_plot[x].append(t[1][1])
			litter_bar_plot_t[x].append(t[1][0])
			#litter_bar_plot_t[x].append(t[4])
			#litter_bar_plot_lit[x].append(t[5])
			figx[x][1].plot(t[2],t[3],'b', label=exp_id)
			#figx[x][1].set_title(exp_id)
			
			#axx.plot(t[2],t[3], C[x], label=exp_id)
		else:
			litter_bar_plot.append([t[1][1]])
			litter_bar_plot_t.append([t[1][0]])
			#ts = t[4]
			#ls = t[5]
			
			#litter_bar_plot_t.append(t[4])
			#litter_bar_plot_lit.append(t[5])
			
			id_list.append(exp_id)
			litter_bar_plot_exps.append('ID:'+exp_id+','+id_map[exp_id][0])
			id_num.append(exp_id)
			
			x = id_list.index(exp_id)
			
			id_c = C_all[c]
			c=c+1
			C.append(id_c)
			figx[x][1].plot(t[2],t[3],'b', label=exp_id)
			figx[x][1].set_title(exp_id)
			figx[x][1].set_xlabel('time in seconds')
			figx[x][1].set_ylabel('quantity of litter')
			
			
			
			#axx.plot(t[2],t[3], id_c, label=exp_id)
	#llg=axx.legend()
	#figx.savefig(folder_name+'litter_collected_TimeStep.pdf', bbox_extra_artists=(llg,), bbox_inches='tight')
	for i in range(len(id_list)):
		ttl = id_list[i]
		#l = figx[i][1].legend()
		
		figx[i][0].savefig(plotsNdata+ttl+fName+'.pdf', bbox_inches='tight')
			
		#litter_bar_plot.append(t[1][1])
		#litter_bar_plot_exps.append('ID:'+exp_id+','+id_map[exp_id][0])
	
	means = []
	std_dev = []
	
	with open(plotsNdata + 'litter_counts'+fName+'.txt','w') as f:
		for v,id_n in zip(litter_bar_plot, id_num):
			std_dev.append(statistics.pstdev(v))
			means.append(statistics.mean(v))
			f.write(id_n + ',' + ','.join([str(i) for i in v]))
			f.write('\n')
		
	num_of_bars = len(litter_bar_plot)
	#means = litter_bar_plot
	#std_dev = [0]*num_of_bars
	
	index = np.arange(num_of_bars)
	bar_width = 0.35
	opacity = 0.4
	error_config = {'ecolor':'0.3'}
	
	#print(litter_bar_plot_exps)
	#print(litter_bar_plot)
	#litter_bar_plot_t = [i[1][0] for i in litter_counts]
	#print(litter_bar_plot_t)
	fig,ax = plt.subplots()
	ax.bar(index,
			means,
			#bar_width,
			alpha=opacity,
			color='b',
			yerr=std_dev,
			error_kw=error_config,
			label=' ')
	plt.title('Total Litter Deposited')#
	#plt.xlabel('Experiments')
	plt.ylabel('Quantity of Litter')#
	#plt.xticks(index,id_num)
	plt.tick_params(
					axis='x',          # changes apply to the x-axis
					which='both',      # both major and minor ticks are affected
					bottom='off',      # ticks along the bottom edge are off
					top='off',         # ticks along the top edge are off
					labelbottom='off') # labels along the bottom edge are off
	m = ['%1.2f' % x for x in means]
	s = ['%1.2f' % x for x in std_dev]
	table = plt.table(cellText=[m,s],
				rowLabels=['mean','std_dev'],
				colLabels=id_num,
				loc='bottom')
	#plt.xticks(index,litter_bar_plot_exps,rotation=90)
	#table.set_fontsize(12)
	#table.scale(5,2)
	lg = plt.legend()
	#plt.tight_layout()
	fig.savefig(plotsNdata+'litter_collected'+fName+'.pdf', bbox_extra_artists=(lg,table,), bbox_inches='tight')
	
	means_t = []	
	std_dev_t = []
		
	with open(plotsNdata + 'time_taken'+fName+'.txt','w') as f:
		for v,id_n in zip(litter_bar_plot_t, id_num):
			std_dev_t.append(statistics.pstdev(v))
			means_t.append(statistics.mean(v))
			f.write(id_n + ',' + ','.join([str(i) for i in v]))
			f.write('\n')
	
	figt,axt = plt.subplots()
	axt.bar(index,
			means_t,
			#bar_width,
			alpha=opacity,
			color='b',
			yerr=std_dev_t,
			error_kw=error_config,
			label=' ')
	plt.title('Simulation time taken')#Total Litter Deposited after 200 seconds
	#plt.xlabel('Experiments')
	plt.ylabel('Time taken')#Quantity of Litter
	#plt.xticks(index,id_num)
	plt.tick_params(
					axis='x',          # changes apply to the x-axis
					which='both',      # both major and minor ticks are affected
					bottom='off',      # ticks along the bottom edge are off
					top='off',         # ticks along the top edge are off
					labelbottom='off') # labels along the bottom edge are off
	m = ['%1.2f' % x for x in means_t]
	s = ['%1.2f' % x for x in std_dev_t]
	table = plt.table(cellText=[m,s],
				rowLabels=['mean','std_dev'],
				colLabels=id_num,
				loc='bottom')
	#plt.xticks(index,litter_bar_plot_exps,rotation=90)
	#table.set_fontsize(12)
	#table.scale(5,2)
	lg = plt.legend()
	#plt.tight_layout()
	figt.savefig(plotsNdata+'litter_collected_t'+fName+'.pdf', bbox_extra_artists=(lg,table,), bbox_inches='tight')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	fig2,ax2 = plt.subplots()
	ind = 0
	xx = range(len(means))
	for b in xx:
		ind = ind + bar_width/2.0
		ax2.bar(ind,
			means[b],
			bar_width/2.0,
			alpha=opacity,
			yerr=std_dev[b],
			error_kw=error_config,
			label=litter_bar_plot_exps[b])
	
	plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    
	lgd = plt.legend(loc=2,bbox_to_anchor=(1,1))
	fig2.savefig(plotsNdata+'litter_collected2'+fName+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
	
	
def max_litter_time():
	#This will plot the time it takes to forage the max litter collected
	#for an algorithm
	return	
	
def litter_collected_10():
	#This will plot bar charts of quantity of litter collected in set of
	#10 time steps.
	return
def litter_vs_time():
	#np_litter_counts = np.array(litter_counts)
	#print('\n'.join([i[0] for i in litter_counts]) )

	#plt.plot(litter_counts[0][1],litter_counts[0][2])
	#plt.show()
	#f = plt.figure()

	#ax = f.add_subplot(5,2,1)
	#ax.plot(litter_counts[0][1],litter_counts[0][2])


	#ax = f.add_subplot(5,2,4)
	#ax.plot(litter_counts[1][1],litter_counts[1][2])
	#f.tight_layout(pad=0.0)
	#f.savefig('test.pdf')
	return
	
def main():
	global tot_lit, lit_pcts
	#my_files = get_experiments_filenames('results/2017_12_19/*')
	my_files = get_experiments_filenames(folder_name)
	readMe = readme[0]
	for p in readMe:
		if 'no_of_lit' in p:
			p = p.split(':')
			tot_lit = float(p[1])
			lit_pcts = pcts * tot_lit
			break
	#print([i[0] for i in robot_files['2017_12_19_13_56_14']])

	#find_index(readme[0])
	#np_readme = np.array(readme)
	#exps = np_readme[:,21]
	#print(exps)
	#PLOT TRAJECTORY OF ALL ROBOTS FOR ALL EXPERIMENTS CONDUCTED.
	robots_trajectory()

	#DO BAR CHART
	litter_collected()
	

	#PLOTTING THE TREND OF LITTER OVER TIME
	litter_vs_time()
	#print(id_map)
	#plt.show()
	
C_all = ['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4',
		 '#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff',
		 '#aa6e28','#fffac8','#800000','#aaffc3','#808000','#ffd8b1',
		 '#000080','#808080','#000000']
my_files = []
robot_files = {}
litter_counts = []
readme = []	
id_map = {}
t2id_map = {}
valid_record = 4
tot_lit = 0# 100 #total litter in world
pcts = 0.9 #percentages expected
lit_pcts = 0# 1 * tot_lit# [tot_lit * p for p in pcts]
fName = '2018-05-19-onecluster5litcapacity'
folder_name = 'results/'+ fName + '/'#'results/2017_12_21-exp1/'
plotsNdata = folder_name + 'plotsNdata/'
#print(len(my_files))

if __name__ == '__main__':
	t0 = time.time()
	pathlib.Path(plotsNdata).mkdir(parents=False,exist_ok=True)
	main()
	print(time.time() - t0)
	
	
