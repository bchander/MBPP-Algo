

### WORKING PROGRAM ........!!!

## NOTE : Adding hit and leave states to the grids list and later using them to fill unoccupied grid lists are very important, since usual way of
# neural network type grid list filling is not suited for most cases esp for mazes n complex envi

'''

## New addition:

Start from goal and move greedily through M-Line, if obstcale is met, then release a new bot, move in either direction of the obstacle,
follow wall until condition A is met
hit point is the point at which the bug originated
leaving point is the pt at which the bug leavs its wall followig behavior whenever it meets M-line

Condition A: 
distance(Goal, hit point) > distance(goal, current point) 
provided that current point is on M-Line

Condition B:
If the bug meets the hit point at which it originated, terminate the bug

Condition C:
If there is no line of sight between the previous vertex or flag point to the current state,
group the neighbors of the current and previous states, note the state that is a common 
neighbor to both of the current and its previous state, which is not in occupied list, and preferably not 
both of them

Condition D:

If the bug meets a leaving point already left by some other bug, then the bug can note all those vertex stored
by the earlier bug which visited that leaving point!

This method also employs "Dynamic Porgramming techniques in which the prob
is divided into sub problems and each subproblem can be used in memory instead of
solving it again. After the find of vertex, DP way of behavior is followed in MBPP

NOTE: for GK
A dynamic programming algorithm will examine the previously solved subproblems and will combine their solutions to give the best solution for the given problem. The alternatives are many, such as using a greedy algorithm
While a greedy algorithm does not guarantee an optimal solution, it is often faster to calculate. Fortunately, some greedy algorithms (such as minimum spanning trees) are proven to lead to the optimal solution.

'''

import numpy as np
import timeit

###-------------------------------------------####

## To find Memory usage ##


# action function --- snew[bot], b = action(s,rr,col,neg_r,m1,m2,m3,x,bot,s1,s2)
def action(s,rr,col,m1,m2,m3,obst,skip,kum):
    k = 0; rr_list =[]; i =1;
    if(obst>0):
        rr[m1/col,m1%col] *= 0.3; rr[m2/col,m2%col] *= 0.3; rr[m3/col,m3%col] *= 0.3;
    while(k ==0):
        rn = s/col; cn = s%col; i = i+1;
        b = [rr[rn,cn+1],rr[rn-1,cn+1],rr[rn-1,cn],rr[rn-1,cn-1],rr[rn,cn-1],rr[rn+1,cn-1],rr[rn+1,cn],rr[rn+1,cn+1]]
        ma = max(b)
        a = [i for i,j in enumerate(b) if j == ma] # greedy action selected with max reward
        if(a[0]==0): snew = s+1;
        elif(a[0]==1): snew = s-(col-1);
        elif(a[0]==2): snew = s-col;
        elif(a[0]==3): snew = s-(col+1);
        elif(a[0]==4): snew = s-1;
        elif(a[0]==5): snew = s+(col-1);
        elif(a[0]==6): snew = s+col;
        elif(a[0]==7): snew = s+(col+1);
        #if (kh==0): ch = snew
        if (obst > 0)or(skip==0): 
            rn = snew/col; cn = snew%col;
            b = [rr[rn,cn+1],rr[rn-1,cn+1],rr[rn-1,cn],rr[rn-1,cn-1],rr[rn,cn-1],rr[rn+1,cn-1],rr[rn+1,cn],rr[rn+1,cn+1]]
            if((kum ==1) and(b.count(-10)>=2))or((kum!=1)and(b.count(-10)>=1)): k=1;
            else: 
                rr[snew/col,snew%col] *= 0.3;
                rr_list.append(snew); k =0;
            '''elif (kum!=1)and(b.count(-10)>2): k=1
                if (bug>0)and(kum ==1)and(snew in [s1[bug-1],s[bug-1]+1,s[bug-1]-1,s[bug-1]+col,s[bug-1]-col]):
                    rr[snew/col,snew%col] *= 0.3; rr_list.append(snew); k =0;
                else: k = 1;'''
        else: k = 1; #break;
        if (i > 7) and (k==0): snew[bug] = m3; k=17; 
    if (obst>0): # val = '%.3f'%(val) -> to round to 3 decimals
        #rr[m1/col,m1%col]/=0.3; rr[m2/col,m2%col]/=0.3; rr[m3/col,m3%col]/=0.3
        t1 = rr[m1/col,m1%col]/0.3; t2 = rr[m2/col,m2%col]/0.3; t3 = rr[m3/col,m3%col]/0.3
        rr[m1/col,m1%col]='%.3f'%(t1); rr[m2/col,m2%col]='%.3f'%(t2); rr[m3/col,m3%col]='%.3f'%(t3)
    for i in range(len(rr_list)):
        #rr[rr_list[i]/col,rr_list[i]%col] /=0.3;
        t = rr[rr_list[i]/col,rr_list[i]%col]/0.3;
        rr[rr_list[i]/col,rr_list[i]%col] = '%.3f'%(t)
    return (snew)

# end of action func

''' RAY-CASTING Algorithm here '''

##########################################
    
def ray_cast(grid1,grid2,col,l_o_s,note): #note is the grid state that is occupied in between found by R_C Algo
    x2 = int(grid2)/col; y2 = int(grid2)%col
    x1 = int(grid1)/col; y1 = int(grid1)%col
    rd = x2 - x1; # Row difference
    cd = y2 - y1; # Column difference
    diff = abs(rd) - abs(cd);
    
    # Some intialization
    x_inc = 1 if (x2>x1) else -1
    y_inc = 1 if (y2>y1) else -1
    n = int(1+ abs(rd) + abs(cd))
    
    # Identifying the two joining lines of the grid based on qurdrant of travel
    if (rd>0 and cd<0) or (rd<0 and cd>0):
        x1_a = x1-0.5; y1_a = y1-0.5; x1_b = x1+0.5; y1_b = y1+0.5;
    if (rd>0 and cd>0) or (rd<0 and cd<0):
        x1_a = x1-0.5; y1_a = y1+0.5; x1_b = x1+0.5; y1_b = y1-0.5;
    if (rd == 0):
        x1_a = x1-0.5; x1_b = x1+0.5; y1_a = y1; y1_b = y1;
    if (cd == 0):
        y1_a = y1-0.5; y1_b = y1+0.5; x1_a = x1; x1_b = x1;
    
    rd_new = abs(rd)*2; cd_new = abs(cd)*2;
    ray_grid_list = [[],[]]
    qu = 2 #if (l_o_s==0) else 1; 
    for j in range(qu):
        x_k = x1; y_k = y1;
        x = x1_a if (j==0) else x1_b;
        y = y1_a if (j==0) else y1_b;
        #diff = abs(rd) - abs(cd);
        for i in range(1,n):
            if (diff > 0):
                diff -= cd_new; x_i = x + x_inc;
                x_j = (x + x_i)/2; # add this to a list of grids
                ray_grid_list[j].append(int(col*x_j + y_k)); # earlier this line was (ray_grid_list.append(col*x_j + y_k));
                ### Be careful !!! dividing with 'col' and again multiplying with 'col' might not result in same value!!!
                x = x_i; x_k = x_j; 
            else :
                diff += rd_new; y_i = y + y_inc;
                y_j = (y + y_i)/2; # add this to a list of grids
                ray_grid_list[j].append(int(col*x_k + y_j)) # earlier this line was (ray_grid_list.append(col*x_k + y_j))
                y = y_i; y_k = y_j;
            ## NEWLY ADDED to note intermediate obstacle point ##
            leng = len(ray_grid_list[j])-1; '''these two lines can be written as  p = ray_grid_list[j][i]'''
            p = ray_grid_list[j][leng]
            if (p in neg_r) and (l_o_s == 0):
                neighb1 = [grid1 +1,grid1 -1,grid1 +col+1,grid1 +col-1,grid1 -col+1,grid1 -col-1,grid1 +col,grid1 -col]
                neighb2 = [grid2 +1,grid2 -1,grid2 +col+1,grid2 +col-1,grid2 -col+1,grid2 -col-1,grid2 +col,grid2 -col]
                #if (p/col != x2) and (p%col != y2):
                if (p/col == x2) or (p%col == y2): note =0;
                elif(p in neighb1)or(p in neighb2): note =0;
                else:
                    note = int(p); i = n+10; j=3; break; # No need for further computation !!!
    if (rd ==1) or (cd==1):
        y = y1; x = x1; count =0;
        occ = 2*(abs(cd)) + 2*(abs(rd))
        for i in range(abs(cd)+1):
            x = x1;
            for j in range(abs(rd)+1):
                count +=1; t = col*x + y;
                if (count <= occ/2):
                    if t not in ray_grid_list[0]: ray_grid_list[0].append(int(t))
                else:
                    if t not in ray_grid_list[1]: ray_grid_list[1].append(int(t)) 
                x = x + x_inc;
            y = y + y_inc;     

    return(ray_grid_list,note);

## END of Ray_Cast Program


''' LINE-OF-SIGHT (LOS-1) Algorithm here '''

''' This code on first round checks from start to goal, then start to goal -1, and so until either there is aLOS or start is reached, once
this round is reached, then the code again starts searching from start to nxt state, start to nxt +1, and so on...'''

def line_of_sight_1(grids,s_s,end_s,col,neg_r):
    ray_grid_list = [[],[]]; l_o_s = 1
    #occupied =[]; state_remove = []
    for i in range(len(grids)):
        current_cell = s_s; j =1;
        ind = 0;
        #next_cell = grids[i][j];
        ze = len(grids[i]);
        check_cell = grids[i][ze-j];
        state_remove = []; #occupied =[]; 
        #neighbours = []
        while(current_cell != grids[i][ze-2]):
            if (current_cell == grids[i][ze-1]): break;
            occupied = []
            check_cell = grids[i][ze-j];
            [ray_grid_list, note] = ray_cast(current_cell,check_cell,col,l_o_s,0);
            for p in (ray_grid_list[0], ray_grid_list[1]):
                occupied += [t for t in p if t in neg_r]
              
            if (len(occupied)<=0): # If walkable (current_cell, check_cell): is true
                #print "check_LOS \n"
                for k in range(ind+1, ze-j):
                    state_remove += [grids[i][k]];
                ind = ze-j;
                current_cell = grids[i][ind];
                j =1;
            else:
                j += 1; ind2 = ze-j; #check_cell = grids[i][ze-j]; 
                if (current_cell == grids[i][ind2-1]):
                    j = 1; current_cell = grids[i][ind2]; ind = ind2            
        if len(state_remove) >0:
            for p in state_remove: grids[i].remove(p);

    return(grids);

### ------------------------------------ ###

## New way from A* implementation

start = timeit.default_timer()

ro = 2 + int(input("  Enter the number of rows of Grid :"))
co = 2 + int(input("  Enter the number of columns of Grid :"))
print("\n")
print("Grid cells are numbered as :")
print("\n")

# To print the states of the environment
'''k = ro
for i in range(1, ro-1):
    print(" ")
    for j in range(1, co-1):
        k += 1
        print(" "), #comma (,) at the end of print statement helps
        print(k),   # in avoiding printing of new line each time
    k+=2
    print("\n")        
'''

print("\n")
print(" ------ From the states indicated above ------ ")
print("\n")
s_s = int(input("  Enter your desired start state:"))
end_s = int(input("  Enter the state of max reward (end state) :"))

print("\n")
print(" Enter the set of index of the states (shown in the above "),
print(" indexed number table) that has puddle in it ")
print(" Leave space between each entered index number, once you "),
print(" are done with entering no's, press ENTER ")
print("\n")

see = raw_input("Enter index list here  : ")
neg_r = [int(i) for i in see.split()] # to get the count of entered inputs
n_k = len(neg_r)

rr = np.zeros((ro,co))
rr[s_s/co,s_s%co] = 1000;

m_line = [s_s]; grids =[[s_s]];
run = 0; col =co; row = ro;
l_o_s = 0; alpha = 0.3
skip = 1; epis =0;
ray_grid_list = [[],[]];

x = end_s/col; y = end_s%col
# heuristic using Manhattan distance
for i in range(1,ro-1):
    for j in range(1,co-1):
        dx = abs(x-i)
        dy = abs(y-j)
        rr[i,j] = int(1000 - (dx + dy))

# print heuristic;

## Choosing M-Line
s =s_s; #action(s[bug],rr,col,mul[bug][0],mul[bug][1],mul[bug][2],k[bug],skip
while end_s not in m_line:
    snew = action(s,rr,co,0,0,0,0,1,0)
    m_line += [snew];
    s = snew;

# identifying the negative states
for i in range(len(neg_r)):
    rr[neg_r[i]/co,neg_r[i]%co] = -10;

###############################################################


# Do MBPP, start from below loop of episode
                                                                 
for episode in range(1,10):
    if (episode >=2) and (skip == 0):
        s=[int(n_ss)]; grids =[[]]; mul = [[0,int(n_ss), int(n_ss)]];
        state_set =[[int(n_ss)]]; leave = [[]]; k = [1]; #hit = [[n_ss]]; 
    else:
        s=[s_s]; grids = [[s_s]]; mul = [[0,s_s, s_s]];
        state_set =[[s_s]]; leave = [[s_s]]; k = [0]; #hit = [[]]; 
    snew=[0];
    hit = [[]]; 
    #c= 0; 
    iterate = 1; bug =0; #ri =0; nb = [1]
    wall = [0]; bha=[0]; #bi =np.zeros(30)
    ri = [0]; kum =[0]; #s_list =[]; ji =[0]; 
    ''' .....Newly added ......'''
    note = 0; l_o_s = 0; #obst = 0; 
    if episode ==1: select_list = [];
    
    for steps in range(0,500):
        bug =0
        while(bug <iterate):
            ''' this is required for this program too !! '''
            if (episode>=2) and (skip == 1):
                if not true_list:
                    new_grids.append(true_list);
                    grids = new_grids; steps = 600; break;
                    
            if ((end_s not in grids[bug])and(0 not in grids[bug]))and(wall[bug]==0): #added latest ("and 0")
                # action(s,rr,col,m1,m2,m3,obst)
                snew[bug] = action(s[bug],rr,col,mul[bug][0],mul[bug][1],mul[bug][2],k[bug],skip,kum[bug])

                if(snew[bug] == end_s): # snew[bot] == end_s
                    grids[bug].append(end_s);

                if(k[bug]>0)and(rr[snew[bug]/col,snew[bug]%col]==0): snew[bug]=s[bug]; wall[bug]+=1
                ''' check if the above condition is req or not ? '''

                if (ri[bug]!=0)and(k[bug]>0): ## CORRECT - works after new bot takes other route
                    t = rr[ri[bug]/col,ri[bug]%col]/alpha; 
                    rr[ri[bug]/col,ri[bug]%col] = '%.3f'%(t); ri[bug] =0;
                if (snew[bug] in mul[bug]): grids[bug] += [-1]; wall[bug]+=1; # To check if the bug meets wall #
                mul[bug][0] = mul[bug][1]; mul[bug][1] = mul[bug][2]; mul[bug][2] = snew[bug];
                rn = snew[bug]/col; cn = snew[bug]%col;
                b = [rr[rn,cn+1],rr[rn-1,cn+1],rr[rn-1,cn],rr[rn-1,cn-1],rr[rn,cn-1],rr[rn+1,cn-1],rr[rn+1,cn],rr[rn+1,cn+1]]
                neighb2 = [s[bug]+1,s[bug]-1,s[bug]+col+1,s[bug]+col-1,s[bug]-col+1,s[bug]-col-1,s[bug]+col,s[bug]-col]
                neighb1 = [snew[bug]+1,snew[bug]-1,snew[bug]+col+1,snew[bug]+col-1,snew[bug]-col+1,snew[bug]-col-1,snew[bug]+col,snew[bug]-col]
                if((end_s not in grids[bug])and(episode<2)and(b.count(-10)>=1)and(k[bug]==1)) or ((episode>=2)and(skip==0)and(k[bug]==1)): ## Latest addition - and(j[bot]==0)
                        k[bug] +=1; kum[bug] += 1
                        #obst += 1; k2[bug] += 1; 
                        wall.append(0); bha.append(0); kum.append(1)
                        k.append(2); # ji.append(0); k2.append(1)
                        '''grids[bug] += [snew[bug]];
                        # removed for now ---> shall be considered for future purpose (in which the waited
                        bugs are assigned based on the list of the hit and leave in the grid list; by assigning all the grid elemnets
                        that are next to the leave/hit point in the grid list as noted  by the hit/leave lists of the specific BUG'''
                        hit[bug] += [s[bug]]; ### Newly added for noting when hit happens 
                        ri.append(snew[bug]);  # added latest to note the cell where new bug is born
                        z1 = len(grids[bug]); grids += [grids[bug][:z1]]
                        #ze = len(state_set[bug]); state_set += [state_set[bug][:ze]]
                        z2 = len(hit[bug]); hit += [hit[bug][:z2]]; # hit point is met
                        z3 = len(leave[bug]); leave += [leave[bug][:z3]]
                        rr[snew[bug]/col,snew[bug]%col] *= alpha ### To scale the reward value of new state for new bot to avoid this
                                                
                        iterate+=1; s.append(s[bug]);snew.append(0); #bi.append(0)
                        #if bug ==0: print s, snew, grids

                        mul.append([0, mul[bug][0],mul[bug][1]]) ## .... Newly changed ....##
                if (k[bug]==0) and (b.count(-10)>=1)and(skip!=0):
                    k[bug] += 1;

                #if bug == 1: print "\nbug status", s[bug], snew[bug], grids, mul, wall

                #state_set[bug] += [snew[bug]]
                if (k[bug]>=2) and (-1 not in grids[bug]): ## LOS algorithm between grid latest vertex and current state ##
                    occuip = []
                    z1 = len(grids[bug])-1;
                    if ((skip != 0)or(skip==0 and len(grids[bug])!=0)): z3 = grids[bug][z1]; '''NEWLY ADDED FOR LOOPING CONDITION'''
                    elif(skip==0 and len(grids[bug])==0): z3 = list1[len(list1)-1];
                    [ray_grid_list, note] = ray_cast(z3,snew[bug],co,l_o_s,note); # l_o_s should be 0
                    for p in (ray_grid_list[0],ray_grid_list[1]):
                        occuip += [t for t in p if t in neg_r];
                    #if snew[bug] == 462: print "check0", bug, grids
                    
                 
                    if (len(occuip)!= 0):
                        if (note != 0):
                            listt1 = [note-col-1,note-col+1,note-col,note-1,note+1];
                            occup = [t for t in listt1 if t in neg_r];
                            if (len(occup)!=0):
                                note2 = note+col+1; note3 = note+col-1;
                            else:
                                note2 = note-col+1; note3 = note-col-1;
                            ''' I'm using the same name : ray_grid_list here, be careful '''
                            ''' Below second time raycast check is definitely needed! - don't delete it'''
                            
                            [ray_grid_list, note4] = ray_cast(z3,note2,co,l_o_s,0);
                            occup = []
                            for p in (ray_grid_list[0],ray_grid_list[1]):
                                occup += [t for t in p if t in neg_r];
                            if (len(occup)== 0):
                                grids[bug] += [int(note2)];
                            else: grids[bug] += [int(note3)];
                            grids[bug] += [snew[bug]]; note = 0;
                            #if snew[bug] == 462: print "check1", bug, grids
                        else: # note the common vertex neighbour for both the new and its previous state
                            if(snew[bug]/col == s[bug]/col)or(snew[bug]%col == s[bug]%col):
                                grids[bug] += [s[bug]]; occup = [s[bug]];
                                #if snew[bug] == 462: print "check2", bug, grids
                            else:
                                #neighb1 = [snew[bug]+1,snew[bug]-1,snew[bug]+col+1,snew[bug]+col-1,snew[bug]-col+1,snew[bug]-col-1,snew[bug]+col,snew[bug]-col]
                                #neighb2 = [s[bug]+1,s[bug]-1,s[bug]+col+1,s[bug]+col-1,s[bug]-col+1,s[bug]-col-1,s[bug]+col,s[bug]-col]
                                occup = [t for t in neighb1 if (t in neighb2 and t not in neg_r)]; ''' BE CAREFUL --- The above line might result in list having multiple states for occup, but occup need only one commmon state - one possible solution is make occup a list of unknown vector size, and keep adding all the common elements to neigh1 and neigh2, then consider one state that among all occup list thta has a neighbour in neg_r'''
                                grids[bug] += [occup[0]];
                                #if snew[bug] == 462: print "at corner", bug, grids
                            if ((episode>=2)and(skip == 0)):
                                ''' This condition wil chk if the selected vertex has LOS with nxt state in list[2], if no no need for further computation'''
                                # Do LOS between snew and list2[0], if LOS exists, stop the bug n wait
                                occuip = []
                                [ray_grid_list, note] = ray_cast(int(occup[0]),int(list2[0]),co,l_o_s,note); # l_o_s should be 0
                                for p in (ray_grid_list[0],ray_grid_list[1]):
                                    occuip += [t for t in p if t in neg_r];
                                if (len(occuip)== 0):
                                    #''' end the bug and add the earlier list states; add this new bug grid list to grids'''
                                    grids[bug] += [0]; leave[bug] += [occup[0]]
                            #if snew[bug] == 462: print "at corner", bug, grids

                if (k[bug]>1)and(skip !=0)and(kum[bug]!=1)and(-1 not in grids[bug]): ## Leaving condition for bug ##
                    '''for i in range(len(leave)):
                        occup += [t for t in leave[i] if t==snew[bug]];'''
                    if (snew[bug] == leave[bug][len(leave[bug])-1]): # this bug meets its own leaving pt from where it is originated
                        grids[bug] += [-1]; ## This bug has to be terminated, no use with this bug
                        wall[bug] = 1; 
                    elif (any(snew[bug] in i for i in leave)): # any() will return true if snew is found in list of leave
                        ''' elif(len(occup)!= 0):''' # chk if bug meets leave point, if met any leave of other bugs, then let it wait
                        #grids[bug] += [snew[bug]]; # Removed for now; shall be considered for future when grid list also include hit/leave points
                        grids[bug] += [0];
                        leave[bug] += [snew[bug]]; i = iterate+1;
                    elif (any(snew[bug] in i for i in hit))and(kum[bug]!=1): # checking if the bug meets alredy visited hit point in m_line
                        grids[bug] += [-1]; ## This bug has to be terminated, no use with this bug
                        wall[bug] = 1;
                    else:
                        if (s[bug] in m_line):
                            if (snew[bug] not in m_line): k[bug] = 0; leave[bug] += [snew[bug]]; snew[bug] = s[bug]; 
                            #chk if prev state is also in m_line and has obst as neigh
                            #elif:(neighb1.count(-10)<=1):
                            '''else:
                                k[bug] = 0; leave[bug] += [snew[bug]];'''
                kum[bug] = 0; # this line is very imp
                '''when bug is generated, then that state is noted in hit list as well as snew, so if kum is not there, then -1 will
                will be added to he grids, which will not allow further iteration'''
                
                #if (skip!=0)and(k[bug]==-1)and(snew[bug] not in m_line): k[bug]=0; leave[bug] = s[bug]; snew[bug] = s[bug];           
                                               
                '''# --------------is this necessary ? ----------------
                if (c!=0): s_list.append(s[bug])
                if (snew[bug] in s_list): grids[bug] += [0]
                # -------------------------------------------
                '''
                
                s[bug] = snew[bug]
                #if snew[bug] == 525: print "\n  checking ......"
            else:
                ''' this condition is very important'''
                bha[bug] = 1
            bug += 1            
        if (bha.count(0) < 1):
                break
    print "\nold grids\n", grids
        
    if (episode <= 10) and (len(grids)>= 2): #removing grids that met wall and removing 0 from incomplete vertex list
        i = 0; nvk = len(grids)
        while (i < nvk):
            if (-1 in grids[i])and(wall[i]==1):
                    grids.remove(grids[i]); nvk -=1
            else: i +=1
        for i in range(len(grids)):
            if (0 in grids[i]): grids[i].remove(0) 
        grids = sorted(grids,key=len) # arrange list according to array len
        grids.reverse() # decreasing order of list len
    nv = len(grids[0])
    kii =0;
    if (episode >=2 and skip ==0):
        ''' condition to add previous list elements to the newly identified vertex list'''
        ti =[[]];
        for i in range(len(grids)):
            for p in range(len(list1)): ti[i] += [list1[p]];
            for j in range(len(grids[i])): ti[i] += [grids[i][j]];
            for q in range(len(list2)): ti[i] += [list2[q]];
            new_grids += [ti[i]];
            if i < (len(grids)-1): ti.append([]);
        grids = new_grids;

    if (episode <= 10) and (len(grids)>= 1):
        ''' Condition to check if path exists or not '''
        for i in range(len(grids)):
            if (end_s in grids[i]): kii+=1;
        if kii ==0:
            print "\n\n","   The environment doesn't have navigabale path !!!"
            print grids
            episode = 100; break;

        # To add vertex/corners - that is not listed for some grid list
        ''' In future work, grid list shall accomodate boith hit and leave points for each bug, such that for every bug that is waiting,
        we can assign subsequent vertex list (including hit/leave points) by making use of the iht/leave points in the grid list.
        we have to add additonal grid elements after that bugs latest hit/leave point'''
        i =0; nvk = len(grids)
        while (i<nvk):
            if(len(grids[i])<nv)and(end_s not in grids[i]):
                for j in range(i):
                    ki = len(grids[i])
                    if (grids[i] != grids[j][:ki]):
                        k = grids[i]+grids[j][ki:]
                        grids.append(k)
                grids.remove(grids[i])
                grids = sorted(grids,key=len)
                grids.reverse()
            i+=1; nvk = len(grids)

        ''' Newly added - checks in a grid list if succesive list elements are same, if so remove that
        and also add extra vertex for the start and goal states such that they allow for robot turns'''

        ## DONT DELETE THE BELOW LINES ....!!!
        '''for i in range(len(grids)): # To remove repeted vertex among the given list
            lee = len(grids[i])
            rem_list = []
            for j in range(len(grids[i])-1): ## remove repeted list elements ##
                if (grids[i][j] == grids[i][j+1]):
                    rem_list += [grids[i][j]];
            for j in range(len(rem_list)):
                grids[i].remove(rem_list[j]);
        '''
        
        ''' ADD Ray-cast and LOS code here - below two lines to remove vertex that has line of sight to its parents parent'''
        grids = line_of_sight_1(grids,s_s,end_s,col,neg_r);
        new_grids = grids;
        
        
    if (episode <= 10):
        ## Now among all the set of available grid lists (diff paths) find the least path
        dist = [];
        #new_grids = grids;
        for i in range(len(grids)):
            dist.append(0)
            x=0; y=0
            for j in range(len(grids[i])-1):
                x = (grids[i][j+1]/col-(grids[i][j])/col)**2 # ** stands for square
                y = (grids[i][j+1]%col-(grids[i][j])%col)**2
                dist[i] += np.sqrt(x+y)
        min_distance = min(dist);    
        min_dist = dist.index(min(dist)) # find the index of the min value of the list, if there are two, will return the first min index
                    
        '''print("\n");
        print ("Choosen grid cell list",grids[min_dist])
        #print("\n","len of grid cell list", len(grids[min_dist]))
        print("\n\n")'''
        
        ## Below two lines to do actual two line R travel, and to adjacent vertex to its neighbours, 
        select_list = grids[min_dist];
        new_grids.remove(select_list);
        for i in range(len(select_list)-1):
            ray_grid_list = [[],[]]; #ray_cast(grid1,grid2,col,l_o_s,note):
            [ray_grid_list,note] = ray_cast(select_list[i],select_list[i+1],col,l_o_s,0);
            occupied_1 = [t for t in ray_grid_list[0] if t in neg_r];
            occupied_2 = [t for t in ray_grid_list[1] if t in neg_r];
            rd = select_list[i+1]/col - select_list[i]/col;
            cd = select_list[i+1]%col - select_list[i]%col
            
            if (len(occupied_1)!=0)and(len(occupied_2)==0):
                l_1 = len(occupied_1);
                if (rd>0 and cd<0): p = occupied_1[0]; n_ss = int(p+col+1); ## selecting the nearest occupied grid cell
                if (rd>0 and cd>0): p = occupied_1[0]; n_ss = int(p+col-1); ## selecting the nearest occupied grid cell
                if (rd<0 and cd>0): p = occupied_1[l_1-1]; n_ss = int(p+col+1); ## selecting the farthest occupied grid cell
                if (rd<0 and cd<0): p = occupied_1[l_1-1]; n_ss = int(p+col-1); ## selecting the farthest occupied grid cell
                ''' nOT CONSIDERED FOR RD =0 AND CD = 0 CASES --- Saved for future works '''
                ki = i+1; ti = select_list[ki:]; j=ki+1;
                select_list.append(0);
                select_list[ki]=int(n_ss);
                for q in range(len(ti)):
                    select_list[j] = ti[q]; j += 1;
                true_list = select_list;
                '''true_list = [select_list[:ki],n_ss,select_list[ki:]];'''
                skip = 1; epis = 17;
                break; ### newly added
                                    
            elif (len(occupied_2)!=0)and(len(occupied_1)==0) :
                l_2 = len(occupied_2);
                if (rd>0 and cd<0): p = occupied_2[l_2-1]; n_ss = int(p-col-1); ## selecting the nearest occupied grid cell
                if (rd>0 and cd>0): p = occupied_2[l_2-1]; n_ss = int(p-col+1); ## selecting the nearest occupied grid cell
                if (rd<0 and cd>0): p = occupied_2[0]; n_ss = int(p-col-1); ## selecting the farthest occupied grid cell
                if (rd<0 and cd<0): p = occupied_2[0]; n_ss = int(p-col+1); ## selecting the farthest occupied grid cell
                ''' nOT CONSIDERED FOR RD =0 AND CD = 0 CASES --- Saved for future works '''
                ki = i+1; ti = select_list[ki:]; j=ki+1;
                select_list.append(0);
                select_list[ki]=int(n_ss);
                for q in range(len(ti)):
                    select_list[j] = ti[q]; j += 1;
                true_list = select_list;        
                skip = 1; epis = 17;
                break; ### newly added
            elif(len(occupied_2)!=0)and(len(occupied_1)!=0):
                p = occupied_1[0];
                neighbours = [p+1,p-1,p-col-1,p-col+1,p+col-1,p+col+1,p+col,p-col]
                unoccupied = [t for t in neighbours if (t not in occupied_1 and t not in occupied_2 and t not in neg_r)]
                n_ss = unoccupied[0];         
                ki = i+1; list1 = select_list[:ki]; list2 = select_list[ki:];
                skip = 0; epis = 17; break; ### newly added
            else:
                if (i == len(select_list)-2):
                    epis = 100;
                    true_list = select_list
                    print "\n", "  Best path is found", "\n","  Navigate through the following states of the grid";
                    print "\n", select_list
                    print "\n", "  Distance of the chosen path is :  ", min_distance 
                    break; ### newly added

    '''# Below lines are just to print the states of the grids
    print ('\n\n')
    print ("---- For episode ---- : ",episode)
    print ('\n')

    print ('\n')
    for i in range(iterate):
        print("  State sequence chosen by bot :  ",i+1)
        print(state_set[i])
        print ('\n')
        #n_steps += len(state_set[i])
        print ("   Number of steps to reach goal :", len(state_set[i]))
        print('\n')
    print "grids" , grids
    for i in range(len(grids)):
        print("  Grid list for bot :",i+1)
        print("\n")
        print(grids[i])
        print("\n\n")
    '''
    if epis == 100: episode = 100; break; # very IMP line don't remove

stop = timeit.default_timer()

print '\n\n Time taken to run this program :'
print stop - start

# To print memory usage
import os
import psutil
process = psutil.Process(os.getpid())
print process.memory_info().rss

#print '\n\n Memory used by the program :',  memory()

###################################3

'''
TO REDUCE COMPUTATION
A. Remove L_O_S, Since in all LOS operation, doube line is considered instead of single line
B. Use same variables wherever possible instead of introducing new variables
C. The for loop with comment "# To remove repeted vertex among the given list" can be removed, it is not needed (check)
D. Rmove heuristic, directly use 'rr'
E.


Adding hit and leave states to the grids list and later using them to fill unoccupied grid lists are very important, since usual way of
neural network type grid list filling is not suited for most cases esp for mazes n complex envi

'''
