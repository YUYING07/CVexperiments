clear;clc;
input_img='lab2.png';
% input_imgΪ����ͼ��output_imgΪ���ͼ��,KΪ˳ʱ����ת�Ƕ�sqrt(1-r^2)
[I,~] = imread(input_img);
input_img = imread(input_img);
%��ȡԭͼ��ĳ���
[width,height,~] = size(I);
for i = 1:width-1
    for j = 1:height-1
        %���Ĺ�һ������
        tempx = (i-0.5*width)/(0.5*width);
        tempy = (j-0.5*height)/(0.5*height);
        %��ȡr��angle
        r  = sqrt(tempx^2 + tempy^2);
        K = (1-r)^2;
        if r >= 1
            x = tempx;
            y = tempy;
        else
            x = cos(K)*tempx - sin(K)*tempy;
            y = sin(K)*tempx + cos(K)*tempy;
        end
        %����ʹ��(uint16()�������д������꣬����ת�����޷���16λ��int���ͣ������������������
        old_x = uint16((x + 1)*0.5*width);
        old_y = uint16((y + 1)*0.5*height);
        %���ͼ��
        output_img(i,j,:) = input_img(old_x,old_y,:);
    end
end
imshow(output_img);
title('���κ��ͼ��');
imwrite(output_img,'shapechange.png');

