clear;clc;
input_img='lab2.png';
% input_img为输入图像，output_img为输出图像,K为顺时针旋转角度sqrt(1-r^2)
[I,~] = imread(input_img);
input_img = imread(input_img);
%获取原图像的长宽
[width,height,~] = size(I);
for i = 1:width-1
    for j = 1:height-1
        %中心归一化坐标
        tempx = (i-0.5*width)/(0.5*width);
        tempy = (j-0.5*height)/(0.5*height);
        %获取r和angle
        r  = sqrt(tempx^2 + tempy^2);
        K = (1-r)^2;
        if r >= 1
            x = tempx;
            y = tempy;
        else
            x = cos(K)*tempx - sin(K)*tempy;
            y = sin(K)*tempx + cos(K)*tempy;
        end
        %必须使用(uint16()函数进行处理坐标，将其转化成无符号16位的int类型，否则坐标索引会出错
        old_x = uint16((x + 1)*0.5*width);
        old_y = uint16((y + 1)*0.5*height);
        %输出图像
        output_img(i,j,:) = input_img(old_x,old_y,:);
    end
end
imshow(output_img);
title('变形后的图像');
imwrite(output_img,'shapechange.png');

