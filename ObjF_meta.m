function D = ...
    ObjF_meta(Fs,Fs_multiclassN,Fs_multiclassR,data_GTT_N,data_GTT_R,label_N,label_R,label,active_set_normal,active_set_rare,numN,numR,a_u)

%Fs = getF(data_p,ParamPR);
num_normal = sum(active_set_normal==1);
num_rare = sum(active_set_rare==1);
sum0 = 0;
sum1 = 0;
sum2 = 0;
sum2N = 0;
sum2R = 0;
sumN = 0;
sumR = 0;
if num_normal>=2
    temp = Fs_multiclassN(label_N~=2,:);
    data_GTT_N = data_GTT_N(label_N~=2,:);
    for i=1:size(temp,1)
        sumN = sumN - log(temp(i,data_GTT_N(i,1)));
    end
    Fn_U = Fs_multiclassN(label_N==2,:);
    tempN = find(active_set_normal==1);
end
if num_rare>=1
    temp = Fs_multiclassR(label_R~=2,:);
    data_GTT_R = data_GTT_R(label_R~=2,:);
    for i=1:size(temp,1)
        sumR = sumR - log(temp(i,data_GTT_R(i,1)));
    end
    Fr_U = Fs_multiclassR(label_R==2,:);
    tempR = find(active_set_rare==1);
end

for i=1:size(Fs,1)
    if label(i)==0 %normal category
        sum0 = sum0-log((1-Fs(i)));
    elseif label(i)==1 %rare category
        sum1 = sum1-log(Fs(i));
    end
end
%sum up the unlabeled samples
Fs_U = Fs(label==2);

%top-layer
for i=1:size(Fs_U,1)
    sum2 = sum2-log(Fs_U(i)*(1-Fs_U(i)));        
end 


if num_normal>=2
    for i=1:size(Fs_U,1)
        for j=1:length(tempN)
            sum2N = sum2N+(1-Fs_U(i))*(log(1/num_normal)-log(Fn_U(i,tempN(j))));
        end        
    end   
end
if num_rare>=1
    for i=1:size(Fs_U,1)    
        for j=1:length(tempR)               
            sum2R = sum2R+Fs_U(i)*(log(1/(1+num_rare))-log(Fr_U(i,tempR(j))));
        end
        %also add up the unknown mass
        sum2R = sum2R+Fs_U(i)*(log(1/(1+num_rare))-log(Fr_U(i,end)));
    end    
end
%make sure sum is used                
if numR==0
    D = (1-a_u)*(sum0+sum1+sumN+sumR)+a_u*(0.5*sum2+sum2N/num_normal+sum2R);
else
    D = (1-a_u)*(sum0+sumN+(numN/numR)*(sum1+sumR))+a_u*(0.5*sum2+sum2N/num_normal+sum2R/(1+num_rare));
end
