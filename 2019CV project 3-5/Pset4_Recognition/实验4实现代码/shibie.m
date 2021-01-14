clear all; 
%读入原始图像
I=imread('timg.jpg'); 
gray=rgb2gray(I); 
ycbcr=rgb2ycbcr(I);%将图像转化为 YCbCr 空间
heighth=size(gray,1);%读取图像尺寸
width=size(gray,2); 
for i=1:heighth %利用肤色模型二值化图像
for j=1:width 
Y=ycbcr(i,j,1); 
Cb=ycbcr(i,j,2); 
Cr=ycbcr(i,j,3); 
if (Y<80) 
gray(i,j)=0; 
else
if (skin(Y,Cb,Cr)==1)%根据色彩模型进行图像二值化
gray(i,j)=255; 
else
gray(i,j)=0; 
end
end
end
end
se=strel('arbitrary',eye(5));%二值图像形态学处理
gray=imopen(gray,se); 
figure;imshow(gray) 
[L,num]=bwlabel(gray,8);%采用标记方法选出图中的白色区域
stats=regionprops(L,'BoundingBox');%度量区域属性
n=1;%存放经过筛选以后得到的所有矩形块
result=zeros(n,4); 
figure,imshow(I); 
hold on; 
for i=1:num %开始筛选特定区域
box=stats(i).BoundingBox; 
x=box(1);%矩形坐标 X
y=box(2);%矩形坐标 Y
w=box(3);%矩形宽度 w
h=box(4);%矩形高度 h
ratio=h/w;%宽度和高度的比例
ux=uint16(x); 
uy=uint8(y); 
if ux>1 
ux=ux-1; 
end
if uy>1 
uy=uy-1; 
end
if w<20 || h<20|| w*h<400 %矩形长宽的范围和矩形的面积可自行设定
continue
elseif ratio<2 && ratio>0.6 && findeye(gray,ux,uy,w,h)==1 
%根据“三庭五眼 ”规则高度和宽度比例应该在（ 0.6,2）内；
result(n,:)=[ux uy w h]; 
n=n+1; 
end
end
if size(result,1)==1 && result(1,1)>0 %对可能是人脸的区域进行标记
rectangle('Position',[result(1,1),result(1,2),result(1,3),result(1,4)],'EdgeColor','r'); 
else
%如果满足条件的矩形区域大于 1,则再根据其他信息进行筛选
a=0; 
arr1=[];arr2=[]; 
for m=1:size(result,1) 
m1=result(m,1); 
m2=result(m,2); 
m3=result(m,3); 
m4=result(m,4); 
%得到符合和人脸匹配的数据
if m1+m3<width && m2+m4<heighth && m3<0.2*width 
a=a+1; 
arr1(a)=m3;arr2(a)=m4; 
%rectangle('Position',[m1,m2,m3,m4],'EdgeColor','r');
end
end
%得到人脸长度和宽度的最小区域
arr3=[];arr3=sort(arr1,'ascend'); 
arr4=[];arr4=sort(arr2,'ascend'); 
%根据得到的数据标定最终的人脸区域
for m=1:size(result,1) 
m1=result(m,1); 
m2=result(m,2); 
m3=result(m,3); 
m4=result(m,4); 
%最终标定人脸
if m1+m3<width && m2+m4<heighth && m3<0.2*width 
m3=arr3(1); 
m4=arr4(1); 
rectangle('Position',[m1,m2,m3,m4],'EdgeColor','r'); 
end
end
end