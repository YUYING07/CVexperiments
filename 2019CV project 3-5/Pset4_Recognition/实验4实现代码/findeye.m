function eye = findeye(bImage,x,y,w,h) 
%FINDEYE Summary of this function goes here
% Detailed explanation goes here
part=zeros(h,w); 
%二值化
for i=y:(y+h) 
for j=x:(x+w) 
if bImage(i,j)==0 
part(i-y+1,j-x+1)=255; 
else
part(i-y+1,j-x+1)=0; 
end
end
end
[L,num]=bwlabel(part,8); 
%如果区域中有两个以上的矩形则认为有眼睛
if num<2 
eye=0; 
else
eye=1; 
end
end 

