import cv2 as cv
import os
from kmeans import kemeans_roi
from Network import getimgfromjson
#from Network import cross_point
import time
import json
import numpy as np
from operator import itemgetter
from yolo import run_yolo_
from yolo_utils import draw_labels_and_boxes
def cross_point(x1,x2,x3,x4,y1,y2,y3,y4):#Calculate center point of parking lots
    k1=(y2-y1)*1.0/(x2-x1)
    b1=y1*1.0-x1*k1*1.0
    if (x4-x3)==0:
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
        y=k1*x*1.0+b1*1.0
    return [x,y] 
# without this line the programme can't run in mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Two categries
#cap = cv.VideoCapture("../RAW_data/Test_video/out.avi")
cap = cv.VideoCapture("./RAW_data/Test_video/out.avi")
# get width and height
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


out = cv.VideoWriter('./RAW_data/demo/demo_video.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

#set frequence of calculation and frequence of logfile output
frequency = 200
frequency1 = 200


# create mask
freespacerange = []
empty_out = []
full_out = []

if(cap.isOpened()==False):
    print("can not read video")

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        #computer time
        start = time.time()

        if(frequency == 200):

            # midfreespace
            boxes, confidences, classids, idxs,colors,labels=run_yolo_(cap)

            #free space with line
            empty_out,full_out,empty_num,full_num = getimgfromjson(frame)

            #reset frequency
            frequency = 0
        else:
            frequency += 1

        #draw freespace
        img22=draw_labels_and_boxes(frame,boxes, confidences, classids, idxs,colors,labels)
        box=sorted(boxes,key=lambda s: s[0])+[[1279,0,1279,0]]
        print(box)
        j=0
        for i in range(len(box)-1):
            if box[i+1][0]>box[i][0]+box[i][2]:#get the free space boundry 
                range_left = box[i][0]+box[1][2]
                range_right = box[i+1][0]
                x_left=(13.32754-13.32859)/1280*(range_left-0)+13.32859
                x_right=(13.32754-13.32859)/1280*(range_right-0)+13.32859

            #figure out area of freespace
                motor = int((range_right - range_left)/20)
                car = int((range_right - range_left)/50)
                cv.putText(frame, "Parking space from"+"52.51303"+","+str(round(x_left,5))+"to"+"52.51303"+","+str(round(x_right,5)) + " engouh for " + str(motor) + " Motorcycle " + str(car)+ " Car", (0, 580 +j*10), cv.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1)
                frame[260:330, range_left:range_right,1] = 255
                frame[260:330, range_left:range_right, 0] = 0
                frame[260:330, range_left:range_right, 2] = 0
                j+=1
        for i in range(len(empty_out)):
            cv.polylines(frame, [empty_out[i]], True, (0, 255, 0))
        for i in range(len(full_out)):
            cv.polylines(frame, [full_out[i]], True, (0, 0, 255))

        ## Set the location of putting text.
        #put text in image
        cv.putText(frame,"Parking_space_with_line: "+str(len(empty_out)),(1050,580),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        
        def harris(image, opt=1):# detect the corner
    # Detector parameters
            blockSize = 2
            apertureSize = 3
            k = 0.04
    # Detecting corners
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            dst = cv.cornerHarris(gray, blockSize, apertureSize, k)
    # Normalizing
            dst_norm = np.empty(dst.shape, dtype=np.float32)
            cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    
            res=[]
    # Drawing a circle around corners
            for i in range(dst_norm.shape[0]):
                for j in range(dst_norm.shape[1]):
                    if int(dst_norm[i, j]) > 120:
                        res.append([j,i])
                        #cv.circle(image, (j, i), 2, (0, 255, 0), 2)
    # output
            res=np.array(res)
            x=res.mean(axis=0)
            print(res)
            print(x)
    
    
    
            return image ,res


        src1 =frame
        src=src1[210:260,60:150,:]

        result ,res= harris(src)
        if res.shape[0]>=4:
            from sklearn.cluster import KMeans
            def kmeans_building(x1,x2,types_num,types,colors,shapes):#divide the corners in to 4 clusters
                X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
                kmeans_model = KMeans(n_clusters=types_num).fit(X)
                x1_result=[]; x2_result=[]
                for i in range(types_num):
                    temp=[]; temp1=[]
                    x1_result.append(temp)
                    x2_result.append(temp1)
                for i, l in enumerate(kmeans_model.labels_):
                    x1_result[l].append(x1[i])
                    x2_result[l].append(x2[i])
    

                return kmeans_model,x1_result,x2_result
            colors = ['b', 'g', 'r','k'] 

            shapes = ['o', 's', 'D','o'] 

            labels=['A','B','C','D']

            kmeans_model,x1_result,x2_result=kmeans_building(res[:,0], res[:,1], 4, labels, colors, shapes) 

        

      

        
            x1=[]
            for i in x1_result:#calculate the average value of each clusters
                i=np.array(i)
                x=i.mean()
                x1.append(x)
            y1=[]
            for j in x2_result:
                j=np.array(j)
                y=j.mean()
                y1.append(y)
      
            x1=np.array(x1)
            y1=np.array(y1)
            x_r=x1+[60,60,60,60]
            y_r=y1+[210,210,210,210]
            x_res=[]
            for i in zip(x_r,y_r):
                x_res.append(list(i))
           
            resu = sorted(x_res)
           
            if (resu[1][0]-resu[0][0])>10 and (resu[2][0]-resu[1][0])>10 and (resu[3][0]-resu[2][0])>10:#make sure the four points are four corners
                x11=(resu[1][0]-88)/88
                x22=(resu[0][0]-70)/70
                x33=(resu[2][0]-112)/112
                x44=(resu[3][0]-131)/131
                y11=(resu[1][1]-232)/232
                y22=(resu[0][1]-257)/257
                y33=(resu[2][1]-256)/256
                y44=(resu[3][1]-232)/232
                for i in resu:
                    cv.circle(frame, (int(i[0]),int(i[1])), 2, (0, 255, 0), 2)
            else:
                x11=0
                x22=0
                x33=0
                x44=0
                y11=0
                y22=0
                y33=0
                y44=0

        path_ofjson = "./RAW_data/josn_file_foto/c5001.json"
    #open josn file
        with open(path_ofjson, 'r') as f:
            locobj = json.load(f)
        dir = locobj['shapes']
        length = len(dir)
        geo=[]
        x0_sum=[]
        for i in range(length):
            if (dir[i]['label'] == 'car'):
                topleft = dir[i]['points'][0]
                if x11>5 or y11>5:#overcome the errors of detection, if difference is more than 5 pixel, the adjustment works
                    topleft=[topleft[0]*x11+topleft[0],topleft[1]*y11+topleft[1]]
                bottomleft = dir[i]['points'][1]
                if x22>5 or y22>5:
                    bottomleft=[bottomleft[0]*x22+bottomleft[0],bottomleft[1]*y22+bottomleft[1]]
                topright = dir[i]['points'][3]
                if x44>5 or y44>5:
                    topright=[topright[0]*x44+topright[0],topright[1]*y44+topright[1]]
                bottomright = dir[i]['points'][2]
                if x33>5 or y33>5:
                    bottomright=[bottomright[0]*x33+bottomright[0],bottomright[1]*y33+bottomright[1]]
                x0=cross_point(topleft[0],bottomright[0],bottomleft[0],topright[0],topleft[1],bottomright[1],bottomleft[1],topright[1])
                
                if (topleft[1]<300):#choose the top parking area and calculate the geo coordinate
                    x_s1=cross_point(10.317460317460318,46.82539682539682, 8.73015873015873,45.23809523809524,165.0793650793651,209.52380952380952,208.73015873015873,164.28571428571428)  
                    x_s2=cross_point(1210.5,1277.5,1258.5,1241.0,168.5,192.5,216.5,168.5)
                    x_c=[round((52.51285-52.51294)/(x_s2[1]-x_s1[1])*(x0[1]-x_s1[1])+52.51294,5),round((13.32747-13.32872)/(x_s2[0]-x_s1[0])*(x0[0]-x_s1[0])+13.32872,5)]
                    cv.putText(frame,str(x_c[0]),(int(bottomleft[0]),int(bottomleft[1]+10)),cv.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)
                    cv.putText(frame,str(x_c[1]),(int(bottomleft[0]),int(bottomleft[1]+20)),cv.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)
                else:#choose the bottom parking area and calculate the geo coordinate
                    x_s1=cross_point(22.22222222222222,80.95238095238095,24.603174603174605,73.80952380952381,438.8888888888889,507.14285714285717,504.76190476190476,439.6825396825397)
                    x_s2=cross_point(1143.1818181818182,1270.4545454545455,1218.1818181818182,1187.5,441.77272727272725,520.1818181818181,522.4545454545455,439.5)
                    x_c=[round((52.51311-52.51317)/(x_s2[1]-x_s1[1])*(x0[1]-x_s1[1])+52.51317,5),round((13.32760-13.32847)/(x_s2[0]-x_s1[0])*(x0[0]-x_s1[0])+13.32847,5)]
                    cv.putText(frame,str(x_c[0]),(int(bottomleft[0]),int(bottomleft[1]+10)),cv.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)
                    cv.putText(frame,str(x_c[1]),(int(bottomleft[0]),int(bottomleft[1]+20)),cv.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)
            geo.append(x_c)
            x0_sum.append(x0)
            
        
        
        for idx,i in enumerate(empty_num):
           
            cv.putText(frame,"free space with line is at: "+str(geo[i][0])+","+str(geo[i][1]),(600,500+10*idx),cv.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)
        outempty=[]
        for i in empty_num:#make the empty parking lots list of logfile
            outempty.append("num"+str(i+1)+"parking lot is free at"+str(geo[i][0])+","+str(geo[i][1]))
        outfull=[]
        for i in full_num:#make the full parking lots list of logfile
            outfull.append("num"+str(i+1)+"parking lot is occupied at"+str(geo[i][0])+","+str(geo[i][1]))
        outdynamic=[]
        
        for i in range(len(box)-1):#make the dynamic parking area information list of logfile
            if box[i+1][0]>box[i][0]+box[i][2]:
                range_left = box[i][0]+box[1][2]
                range_right = box[i+1][0]
                x_left=(13.32754-13.32859)/1280*(range_left-0)+13.32859
                x_right=(13.32754-13.32859)/1280*(range_right-0)+13.32859

            #figure out area of freespace
                motor = int((range_right - range_left)/20)
                car = int((range_right - range_left)/50)
                outdynamic.append("Parking space from"+"52.51303"+","+str(round(x_left,5))+"to"+"52.51303"+","+str(round(x_right,5)) + " enough for " + str(motor) + " Motorcycle " + str(car)+ " Car")
        if frequency1 == 200:
            with open("out.txt","w",encoding='utf-8') as f:
                for i in outempty:
                    f.writelines(i+'\n')
                for j in outfull:
                    f.writelines(j+'\n')
                for k in outdynamic:
                    f.writelines(k+'\n')
            frequency1 =0
        else:
            frequency1 +=1
        
            
            
            
        end=time.time()
        print("used:",end-start)

        #cv.imshow("frame", frame)
        #cv.waitKey(25)
        #out.write(frame)

    # Break the loop
    else:
        break