import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3,640)  #3 -> genişlik , 4 -> yüksekliktir. çözünürlüğü 640*480 olarak ayarladık.
cap.set(4,480)

mpHand = mp.solutions.hands #mp içindeki el izleme modülü olan hands modülünü yükledik.
hands = mpHand.Hands()  # el tanıma modülünü kullanmak için ahns() fonksiyonu ekler. 
mpDraw = mp.solutions.drawing_utils #elin üzerindeki eklemleri çizdirme işlemi


tipIds =[4,8,12,16,20] #parmağın tepe uç noktalarının kordinatları.
while True:
    
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB) #el tespit ettiği zaman konumlandırma alır.
    #print(results.multi_hand_landmarks)
    
 #el iskeleti oluşturma  
    lmList = [] #liste oluşturduk lmlerin içine depolanması için daha sonra id ve kordinatları bunun içine ekleyeceğiz. 
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHand.HAND_CONNECTIONS)
           
 # eklemlerin id ve x,y,z notalarını bulma
        for id, lm in enumerate (handLms.landmark):
            h, w, _ = img.shape # yükseklik, genişlik ve renk döndürür. renk kullanmayacaksak c yerine _ yazarız.
            cx,cy =int(lm.x*w), int(lm.y*h)
            lmList.append([id,cx,cy]) #idleri ve lm noktalarını liste içine ekledik.
            
            
    if len(lmList) != 0:
        fingers =[]
 
        #baş parmak için;// sağ kapanırsa 0 sola açılırsa 1
        if lmList[tipIds[0]] [1] < lmList[tipIds[0] - 1 ] [1]: #soldaki bağ parmağın dikey y eksen olduğu
            fingers.append(1)
        else:
            fingers.append(0)
            
        #diğer 4 parmak // en tepe uç orta ucun altına gelirse kapanmış olur.   
        for id in range(1,5):
            if lmList[tipIds[id]] [2] < lmList[tipIds[id] - 2] [2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
         #print(fingers)       
                
        totalF = fingers.count(1)  #listede kaç tane 1 açık parmak olduğu hesabını yapar.
        #print(totalF)
        
        cv2.putText(img, str(totalF), (30,125), cv2.FONT_HERSHEY_PLAIN,10, (255,0,0),8)
                 
   # print(lmList) # consol kısmında matris şeklinde eklem idsi ve x y kordinatlarını gördük.

    cv2.imshow("parmak sayma",img)
    cv2.waitKey(1)