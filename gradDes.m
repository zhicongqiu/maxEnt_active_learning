function [ParamPR_old ParamN_old ParamR_old] = gradDes(data_raw,data_p,data_GTT,label,a_u,ParamPR_old,ParamN_old,ParamR_old,active_set_normal,active_set_rare,numN,numR,pval)

%gradient descent
%note: gradient updates can be parallelized for faster computation

%%%%%%%%%%%%%%%%
%initial step size and tolerance level
step_size = 1e-4;
reltol = 1e-4;

num_active_normal = sum(active_set_normal);
num_active_rare = sum(active_set_rare);

%initialize inter-normal and inter-rare
data_GTT_tempN = 0;
data_GTT_tempR = 0;
data_tempN = 0;
data_tempR = 0;
label_tempN = 0;
label_tempR = 0;
ParamN_new = 0;
ParamR_new = 0;
if num_active_normal>=2
    tempN = find(active_set_normal==1);
    %first,filter out rare samples
    data_tempN = data_raw(label~=1,:);
    data_GTT_tempN = data_GTT(label~=1,:);
    label_tempN = label(label~=1);
    ParamN_new = ParamN_old;
end
if num_active_rare>=1
    tempR = find(active_set_rare==1);
    %first,filter out rare samples
    data_tempR = data_raw(label~=0,:);
    data_GTT_tempR = data_GTT(label~=0,:);
    label_tempR = label(label~=0);
    ParamR_new = ParamR_old;
end

%normal/rare
%initialize
ParamPR_new = ParamPR_old;
Fs_multiclassN = 0;
Fs_multiclassR = 0;
Fs = getF(data_p,ParamPR_old);
if num_active_normal>=2
    Fs_multiclassN = getF_multiclassNormal(data_tempN,data_GTT_tempN,label_tempN,ParamN_old,active_set_normal);
end
if num_active_rare>=1
    Fs_multiclassR = getF_multiclassRare(data_tempR,data_GTT_tempR,label_tempR,ParamR_old,active_set_rare);
end 

Dold =  ObjF_meta(Fs,Fs_multiclassN,Fs_multiclassR,data_GTT_tempN,data_GTT_tempR,label_tempN,label_tempR,label,active_set_normal,active_set_rare,numN,numR,a_u);

Dnew = inf;
updated = true;
mu = step_size;
count_step = 0;
while abs(Dnew-Dold)>=reltol||Dnew>Dold||isinf(Dold) 

    count_step = count_step+1;
    if updated==true
        temp_PR = 0;
        temp_normal = 0;
        temp_rare = 0;
        Fs = getF(data_p,ParamPR_old);
        if num_active_normal>=2
            Fs_multiclassN = getF_multiclassNormal(data_tempN,data_GTT_tempN,label_tempN,ParamN_old,active_set_normal);
        end
        if num_active_rare>=1
            Fs_multiclassR = getF_multiclassRare(data_tempR,data_GTT_tempR,label_tempR,ParamR_old,active_set_rare);
        end        
        
        Dalpha0 = get_Dalpha(ones(size(data_p,1),1),Fs,Fs_multiclassN,Fs_multiclassR,...
				  label_tempN,label_tempR,label,active_set_normal,active_set_rare,numN,numR,a_u);        
        for i=1:size(data_p,2)
            Dalpha(i) = get_Dalpha(-1*data_p(:,i),Fs,Fs_multiclassN,Fs_multiclassR,...
						label_tempN,label_tempR,label,active_set_normal,active_set_rare,numN,numR,a_u);
        end
        temp_PR = temp_PR+Dalpha0^2+norm(Dalpha)^2;
        
        if num_active_normal>=2
            %update parameters for each class           
            %initialize Dbeta0 and Dbeta
            Dbeta0N = zeros(length(tempN)-1,1);
            DbetaN = zeros(length(tempN)-1,size(data_raw,2));           
            for i=1:length(tempN)-1
                Dbeta0N(i) = ...
                    get_Dbeta(ones(size(data_tempN,1),1),1-Fs(label~=1),Fs_multiclassN,data_GTT_tempN,label_tempN,tempN(i),active_set_normal,num_active_normal,numN,numR,a_u,false);
                temp_normal = temp_normal+Dbeta0N(i)^2;
                for j=1:size(data_raw,2)
                    DbetaN(i,j) = ...
                        get_Dbeta(data_tempN(:,j),1-Fs(label~=1),Fs_multiclassN,data_GTT_tempN,label_tempN,tempN(i),active_set_normal,num_active_normal,numN,numR,a_u,false);
                end
                temp_normal = temp_normal+norm(DbetaN(i,:))^2;
            end
        end            
        if num_active_rare>=1
            %update parameters for each class           
            %initialize Dbeta0 and Dbeta
            Dbeta0R = zeros(length(tempR),1);
            DbetaR = zeros(length(tempR),size(data_raw,2));           
            for i=1:length(tempR)
                Dbeta0R(i) = ...
                    get_Dbeta(ones(size(data_tempR,1),1),Fs(label~=0),Fs_multiclassR,data_GTT_tempR,label_tempR,tempR(i),active_set_rare,num_active_rare,numN,numR,a_u,true);
                temp_rare = temp_rare+Dbeta0R(i)^2;
                for j=1:size(data_raw,2)
                    if pval==false
                        DbetaR(i,j) = ...
                            get_Dbeta(data_tempR(:,j),Fs(label~=0),Fs_multiclassR,data_GTT_tempR,label_tempR,tempR(i),active_set_rare,num_active_rare,numN,numR,a_u,true);
                    else
                        DbetaR(i,j) = ...
                            get_Dbeta(-1*data_tempR(:,j),Fs(label~=0),Fs_multiclassR,data_GTT_tempR,label_tempR,tempR(i),active_set_rare,num_active_rare,numN,numR,a_u,true);

                    end
                end
                temp_rare = temp_rare+norm(DbetaR(i,:))^2;
            end
        end                    
        temp_normALL = norm([temp_PR temp_normal temp_rare]); 
        %disp(temp_normALL);
        if count_step>1
            Dold = Dnew;
        end
    end
    ParamPR_new.alpha0 = ParamPR_old.alpha0-mu*Dalpha0;%/norm(temp_PR);
    ParamPR_new.alpha = ParamPR_old.alpha-mu.*Dalpha;%./norm(temp_PR);   
    %positive constraints, projected gradient
    ParamPR_new.alpha(ParamPR_new.alpha<0) = 0;
    
    %inter-normal update
    if num_active_normal>=2
        for i=1:length(tempN)-1
            ParamN_new(tempN(i)).beta0 = ParamN_old(tempN(i)).beta0-mu*Dbeta0N(i);%/norm(temp_normal);
            %unconstraint weights
            ParamN_new(tempN(i)).beta = ParamN_old(tempN(i)).beta-mu*DbetaN(i,:);%./norm(temp_normal);
        end
    end
    %inter-rare update
    if num_active_rare>=1
        for i=1:length(tempR)
	     
            ParamR_new(tempR(i)).beta0 = ParamR_old(tempR(i)).beta0-mu*Dbeta0R(i);%/norm(temp_rare);
            %unconstraint weights
            ParamR_new(tempR(i)).beta = ParamR_old(tempR(i)).beta-mu*DbetaR(i,:);%./norm(temp_rare);
            if pval==true
                ParamR_new(tempR(i)).beta(ParamR_new(tempR(i)).beta<0) = 0;
            end
        end
    end    
    
    Fs = getF(data_p,ParamPR_new);
    if num_active_normal>=2
        Fs_multiclassN = getF_multiclassNormal(data_tempN,data_GTT_tempN,label_tempN,ParamN_new,active_set_normal);
    end
    if num_active_rare>=1
        Fs_multiclassR = getF_multiclassRare(data_tempR,data_GTT_tempR,label_tempR,ParamR_new,active_set_rare);
    end     
    Dnew = ObjF_meta(Fs,Fs_multiclassN,Fs_multiclassR,data_GTT_tempN,data_GTT_tempR,label_tempN,label_tempR,label,active_set_normal,active_set_rare,numN,numR,a_u);

    if Dnew<Dold
        updated = true;
        mu = step_size;
        ParamPR_old = ParamPR_new;
        if num_active_normal>=2
            ParamN_old = ParamN_new;
        end
        if num_active_rare>=1
            ParamR_old = ParamR_new;
        end
    else
        updated = false;
        %reduce step size
        mu = 0.5*mu;
    end
        
end
fprintf('end of GD in %d steps, with Dold = %g\n',count_step,Dold);
end
    
    