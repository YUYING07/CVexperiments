clear all; 
%����ԭʼͼ��
I=imread('timg.jpg'); 
gray=rgb2gray(I); 
ycbcr=rgb2ycbcr(I);%��ͼ��ת��Ϊ YCbCr �ռ�
heighth=size(gray,1);%��ȡͼ��ߴ�
width=size(gray,2); 
for i=1:heighth %���÷�ɫģ�Ͷ�ֵ��ͼ��
for j=1:width 
Y=ycbcr(i,j,1); 
Cb=ycbcr(i,j,2); 
Cr=ycbcr(i,j,3); 
if (Y<80) 
gray(i,j)=0; 
else
if (skin(Y,Cb,Cr)==1)%����ɫ��ģ�ͽ���ͼ���ֵ��
gray(i,j)=255; 
else
gray(i,j)=0; 
end
end
end
end
se=strel('arbitrary',eye(5));%��ֵͼ����̬ѧ����
gray=imopen(gray,se); 
figure;imshow(gray) 
[L,num]=bwlabel(gray,8);%���ñ�Ƿ���ѡ��ͼ�еİ�ɫ����
stats=regionprops(L,'BoundingBox');%������������
n=1;%��ž���ɸѡ�Ժ�õ������о��ο�
result=zeros(n,4); 
figure,imshow(I); 
hold on; 
for i=1:num %��ʼɸѡ�ض�����
box=stats(i).BoundingBox; 
x=box(1);%�������� X
y=box(2);%�������� Y
w=box(3);%���ο�� w
h=box(4);%���θ߶� h
ratio=h/w;%��Ⱥ͸߶ȵı���
ux=uint16(x); 
uy=uint8(y); 
if ux>1 
ux=ux-1; 
end
if uy>1 
uy=uy-1; 
end
if w<20 || h<20|| w*h<400 %���γ���ķ�Χ�;��ε�����������趨
continue
elseif ratio<2 && ratio>0.6 && findeye(gray,ux,uy,w,h)==1 
%���ݡ���ͥ���� ������߶ȺͿ�ȱ���Ӧ���ڣ� 0.6,2���ڣ�
result(n,:)=[ux uy w h]; 
n=n+1; 
end
end
if size(result,1)==1 && result(1,1)>0 %�Կ�����������������б��
rectangle('Position',[result(1,1),result(1,2),result(1,3),result(1,4)],'EdgeColor','r'); 
else
%������������ľ���������� 1,���ٸ���������Ϣ����ɸѡ
a=0; 
arr1=[];arr2=[]; 
for m=1:size(result,1) 
m1=result(m,1); 
m2=result(m,2); 
m3=result(m,3); 
m4=result(m,4); 
%�õ����Ϻ�����ƥ�������
if m1+m3<width && m2+m4<heighth && m3<0.2*width 
a=a+1; 
arr1(a)=m3;arr2(a)=m4; 
%rectangle('Position',[m1,m2,m3,m4],'EdgeColor','r');
end
end
%�õ��������ȺͿ�ȵ���С����
arr3=[];arr3=sort(arr1,'ascend'); 
arr4=[];arr4=sort(arr2,'ascend'); 
%���ݵõ������ݱ궨���յ���������
for m=1:size(result,1) 
m1=result(m,1); 
m2=result(m,2); 
m3=result(m,3); 
m4=result(m,4); 
%���ձ궨����
if m1+m3<width && m2+m4<heighth && m3<0.2*width 
m3=arr3(1); 
m4=arr4(1); 
rectangle('Position',[m1,m2,m3,m4],'EdgeColor','r'); 
end
end
end