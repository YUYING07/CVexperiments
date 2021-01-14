clear;clc;
% input_imgΪ����ͼ��output_imgΪ���ͼ��,x,yΪ���ű���
input_img='lab2.png';
x=1.8;
y=0.8;

[I,map] = imread(input_img);
input_img = imread(input_img);
%��ȡԭͼ��ĳ����ά��
[width,height,dimension] = size(I);

%������ͼ��ĳ���
%��������ʹȡ�õ���ֵΪ����
new_width = round(width*x);
new_height = round(height*y);

%˫���Բ�ֵ��
for i = 1:new_width
    for j = 1:new_height
        %ԭ��ͼ�������,����ȡ��
        tempx = floor((i-1)/x);
        tempy = floor((j-1)/y);
        %�������ߺ��ĸ�������д���
        if tempx == 0 || tempy == 0 || tempx == width-1 || tempy == height-1
            output_img(1,j,:) = input_img(1,tempy+1,:);
            output_img(i,1,:) = input_img(tempx+1,1,:);
        %���������ؽ��д���
        else
            %����ԭͼ�������ȥ��ͼ�������С������
            a = (i-1) / x - tempx;
            b = (j-1) / y - tempy;
            %��СֵΪ1
            tempx = tempx+1;
            tempy = tempy+1;
            output_img(i,j,:) = input_img(tempx,tempy,:)*(1-a)*(1-b)+input_img(tempx,tempy+1,:)*(1-a)*b...
                +input_img(tempx+1,tempy,:)*a*(1-b)+input_img(tempx+1,tempy+1,:)*a*b;
        end
    end
end
figure();
imshow(output_img);
title(['�任���ͼ�񣨴�С�� ',num2str(new_width),'*',num2str(new_height),'*',num2str(dimension)',')']);
imwrite(output_img,'sizechange.png');
