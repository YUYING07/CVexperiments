%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%利用提供的两幅立体图像进行视觉立体匹配操作，
%%要求实现获得该图像的视差图像以及视差图的深度图，
%%并将深度图用3D视图展示；
%%时间：2019/1/26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
tic
% User can set DisparityRange and matching variable.
%You can set upper bound of disparity search range by dmax
dmax=40;
%Select window type for calculation error energy
% matching=1 for point, matching=2 for line, matching=3 for 3x3 square
% window
matching=3;

% Alfa tolerance coefficient for eliminating unreliable estimation
Alfa=1;

% Stereo Camera System Parameter for dept map calcualtion
% foc: focal length of cameras in unit of cm
foc=30;
% T: distance between stereo camera pair in unit of cm
T=20;
%Loading Right image to XR and Left iamage to XL
[XR,MAP] = imread('E:\matlab\shijue\shiyan4\view5m.png');
[XL,MAP] = imread('E:\matlab\shijue\shiyan4\view1m.png');

%------------------ END of USER SETTINGS----------------------

% auto-presettings for the program
[m n p]=size(XR);
Edis=1000000*ones(m,n);
disparity=zeros(m,n);
XR=double(XR);
XL=double(XL);

% process by increasing disparity
for d=0:dmax
    fprintf ('计算差异: %d\n',d);
% composing error energy matrix for every disparity.(Algorithm step:1)
for j=3+d:n-2
    for i=2:m-1
         if p==3
           %kareselfark(i,j-d)=(XR(i,j,1)-XL(i,j-d,1))^2+(XR(i,j,2)-XL(i,j-d,2))^2+(XR(i,j,3)-XL(i,j-d,3))^2;  
           if matching==1
           %point matching 
            ErrorEnergy(i,j-d)=(1/3)*[(XL(i,j,1)-XR(i,j-d,1))^2+(XL(i,j,2)-XR(i,j-d,2))^2+(XL(i,j,3)-XR(i,j-d,3))^2];
           elseif matching==2 
           %block matching with line type window
            ErrorEnergy(i,j-d)=(1/15)*[(XL(i,j,1)-XR(i,j-d,1))^2+(XL(i,j,2)-XR(i,j-d,2))^2+(XL(i,j,3)-XR(i,j-d,3))^2+(XL(i,j-1,1)-XR(i,j-1-d,1))^2+(XL(i,j-1,2)-XR(i,j-1-d,2))^2+(XL(i,j-1,3)-XR(i,j-1-d,3))^2+(XL(i,j+1,1)-XR(i,j+1-d,1))^2+(XL(i,j+1,2)-XR(i,j+1-d,2))^2+(XL(i,j+1,3)-XR(i,j+1-d,3))^2+(XL(i,j-2,1)-XR(i,j-2-d,1))^2+(XL(i,j-2,2)-XR(i,j-2-d,2))^2+(XL(i,j-2,3)-XR(i,j-2-d,3))^2+(XL(i,j+2,1)-XR(i,j+2-d,1))^2+(XL(i,j+2,2)-XR(i,j+2-d,2))^2+(XL(i,j+2,3)-XR(i,j+2-d,3))^2];
           else
               top=0;
               for k=i-1:i+1
                   for l=j-1:j+1
                     top=top+(XL(k,l,1)-XR(k,l-d,1))^2+(XL(k,l,2)-XR(k,l-d,2))^2+(XL(k,l,3)-XR(k,l-d,3))^2;  
                   end
               end
               ErrorEnergy(k,l-d)=(1/27)*top;
           end 
       else
           Disp('错误警告：将RGB彩色图像用于立体声对!!');
        end
    end
end
% applying smooting on error energy surfaces by iterative averaging
% filtering. (Algorithm step:2)
ErrorEnergyFilt=IterativeAveragingFilter(ErrorEnergy,1,[4 4]);

% selecting disparity which has minimum error energy.(Algorithm step:3)
[m1 n1]=size(ErrorEnergyFilt);
for k=1:m1
    for l=1:n1
       if Edis(k,l)>ErrorEnergyFilt(k,l)
            disparity(k,l)=d;
            Edis(k,l)=ErrorEnergyFilt(k,l);
        end        
    end
end
end
% clear 1000000 pre-setting in Edis
for k=1:m
    for l=1:n
       if Edis(k,l)==1000000
            Edis(k,l)=0;
        end        
    end
end

% extracting calculated zone
nx=n-dmax;
for k=2:m-1
    for l=2:nx-1
        disparityx(k,l)=disparity(k,l);
        %Edisx(k,l)=Edis(k,l);
        %regMapx(k,l)=regMap(k,l);
        XLx(k,l)=XL(k,l);
        XRx(k,l)=XR(k,l);
        top=0;
        for x=k-1:k+1
            for y=l-1:l+1
                top=top+(XL(x,y+disparity(k,l),1)-XR(x,y,1))^2+(XL(x,y+disparity(k,l),2)-XR(x,y,2))^2+(XL(x,y+disparity(k,l),3)-XR(x,y,3))^2;  
            end
        end
        Ed(k,l)=(1/27)*top;
    end
end

%calculates error energy treshold for reliablity of disparity
Toplam=0;
for k=1:m-1
    for l=1:nx-1
      Toplam=Toplam+Ed(k,l);       
    end
end
% Error threshold Ve
Ve=Alfa*(Toplam/((m-1)*(nx-1)));

EdReliable=Ed;
disparityReliable=disparityx;
Ne=zeros(m,nx);
for k=1:m-1
    for l=1:nx-1
       if Ed(k,l)>Ve
          % sets unreliable disparity to zero
          disparityReliable(k,l)=0;
          EdReliable(k,l)=0;
          Ne(k,l)=1; % indicates no-estimated state
        end        
    end
end

% calculating reliablities both raw disparity and filtered disparity
TopE=0;
TopER=0;
Sd=0;
for k=1:m-1
    for l=1:nx-1
          TopE=TopE+Ed(k,l);
          if Ne(k,l)==0          
             TopER=TopER+EdReliable(k,l);
             Sd=Sd+1;
          end          
    end
end
ReliablityE=((nx-1)*(m-1))/(TopE);
ReliablityER=(Sd)/(TopER);

% median filtering for repairment of occulations
%disparityF=IterativeAveragingFilter(disparity,5,[4 4]);
disparityF=medfilt2(disparityReliable,[5 5]);

for k=1:m-1
    for l=1:nx-1
          % Zero disparity produce zero dept
          if disparityF(k,l)<5;
              DepthMap(k,l)=0;
          else
              DepthMap(k,l)=foc*(T/disparityF(k,l));
        end        
    end
end

fprintf ('******** 可靠性报告  ********** \n')
fprintf ('视差图的可靠性: %f \n',ReliablityE)
fprintf ('过滤的视差图的可靠性: %f \n',ReliablityER)
fprintf ('******** Algoritm速度报告  ********** \n')
fprintf ('时间用于计算: %f \n',toc)

figure(1)
imagesc(disparityx);colorbar;
colormap('gray')
title('视差图')

figure(2)
colormap('gray')
imagesc(disparityReliable);colorbar;
title('具有可靠差异的视差图')


figure(3)
colormap('gray')
imagesc(disparityF);colorbar;
title('具有可靠差异的中值滤波视差图')


figure(4)
colormap('gray')
imagesc(DepthMap);colorbar;
title('具有可靠差异的视差图的深度图[cm]')


figure(5)
colormap('gray')
imagesc(log10(Ed));colorbar;
title('差异映射误差能量')


figure(6)
imagesc(XR./255)
title('右相机彩色图像')

figure(7)
imagesc(XL./255)
title('左相机彩色图像')


figure(8)
colormap('bone')
mesh(disparityF)
title('3D视图')