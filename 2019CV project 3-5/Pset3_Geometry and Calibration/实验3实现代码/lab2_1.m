clear;clc;
% input_img为输入图像，output_img为输出图像,x,y为缩放倍数
input_img='lab2.png';
x=1.8;
y=0.8;

[I,map] = imread(input_img);
input_img = imread(input_img);
%获取原图像的长宽和维度
[width,height,dimension] = size(I);

%计算新图像的长宽
%四舍五入使取得的新值为整数
new_width = round(width*x);
new_height = round(height*y);

%双线性差值法
for i = 1:new_width
    for j = 1:new_height
        %原来图像的坐标,向下取整
        tempx = floor((i-1)/x);
        tempy = floor((j-1)/y);
        %对四条边和四个顶点进行处理
        if tempx == 0 || tempy == 0 || tempx == width-1 || tempy == height-1
            output_img(1,j,:) = input_img(1,tempy+1,:);
            output_img(i,1,:) = input_img(tempx+1,1,:);
        %对其余像素进行处理
        else
            %计算原图像坐标减去新图像坐标的小数部分
            a = (i-1) / x - tempx;
            b = (j-1) / y - tempy;
            %最小值为1
            tempx = tempx+1;
            tempy = tempy+1;
            output_img(i,j,:) = input_img(tempx,tempy,:)*(1-a)*(1-b)+input_img(tempx,tempy+1,:)*(1-a)*b...
                +input_img(tempx+1,tempy,:)*a*(1-b)+input_img(tempx+1,tempy+1,:)*a*b;
        end
    end
end
figure();
imshow(output_img);
title(['变换后的图像（大小： ',num2str(new_width),'*',num2str(new_height),'*',num2str(dimension)',')']);
imwrite(output_img,'sizechange.png');
