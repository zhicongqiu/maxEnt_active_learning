function [error FA FN error_CN error_CR standard_error_avgC unknown_error_avgC]= calculate_error(test_PMFunknown,test_PMFnormal,test_PMFrare,test_GTT,active_set_rare,count)

%overall classification error
M = size(test_PMFunknown,1);

error = 0;
error_CN = 0;
error_CR = 0;
FA = 0;
FN = 0;

error_normalPC= zeros(1,size(test_PMFnormal,2));

error_rarePC= zeros(1,size(test_PMFrare,2)-1);
standard_error_rarePC= zeros(1,size(test_PMFrare,2)-1);
unknown_error_rarePC= zeros(1,size(test_PMFrare,2)-1);

total_normalPC = zeros(1,size(test_PMFnormal,2));
total_rarePC = zeros(1,size(test_PMFrare,2)-1);


total_normal = sum(test_GTT(:,2)==0);
total_rare = M-total_normal;
standard_error_avgC = 0;
unknown_error_avgC = 0;

for i=1:M
    %determine its belonging based on inference
    [an bn] = max((1-test_PMFunknown(i))*test_PMFnormal(i,:));
    if sum(active_set_rare)==0
        ar = test_PMFunknown(i);
        br = size(test_PMFrare,2);
    else
        [ar br] = max(test_PMFunknown(i)*test_PMFrare(i,:));
    end
    if test_GTT(i,2)==0
        total_normalPC(test_GTT(i,1)) = total_normalPC(test_GTT(i,1))+1;
        if an<ar  %normal sample error
            FA = FA+1;
            error = error+1;
            %error_CN = error_CN+1;
            error_normalPC(test_GTT(i,1)) = error_normalPC(test_GTT(i,1))+1;
        elseif test_GTT(i,1)~=bn %inter-normal class error
            error_CN = error_CN+1;
            error = error+1;
            error_normalPC(test_GTT(i,1)) = error_normalPC(test_GTT(i,1))+1;
        end                
    else
        total_rarePC(test_GTT(i,1)) = total_rarePC(test_GTT(i,1))+1;
        if an>ar %rare sample error
            FN = FN+1;
            error = error+1;
            error_rarePC(test_GTT(i,1)) = error_rarePC(test_GTT(i,1))+1;
        elseif active_set_rare(test_GTT(i,1))==0     
            %standard error calculation...
            error_CR = error_CR+1;
            error = error+1;
            standard_error_rarePC(test_GTT(i,1)) = standard_error_rarePC(test_GTT(i,1))+1; 
            if br~=size(test_PMFrare,2)
                %updated error calculation...
                unknown_error_rarePC(test_GTT(i,1)) = unknown_error_rarePC(test_GTT(i,1))+1; 
            end
        elseif test_GTT(i,1)~=br
            error_CR = error_CR+1;
            error = error+1;
            error_rarePC(test_GTT(i,1)) = error_rarePC(test_GTT(i,1))+1;                
        end
    end        
end

if count == 0 %report error rate on test set
    if M~=0
        error = error/M;
    else
        error = 0;
    end       
    if total_normal~=0
        error_CN = error_CN/total_normal;
        FA = FA/total_normal;
    else
        error_CN = 0;
        FA = 0;
    end
    if total_rare~=0
        error_CR = error_CR/total_rare;
        FN = FN/total_rare;
    else
        error_CR = 0;
        FN = 0;
    end
    standard_error_avgC = ...
        mean([error_normalPC(total_normalPC~=0)./total_normalPC(total_normalPC~=0) ...
        (standard_error_rarePC(total_rarePC~=0)+error_rarePC(total_rarePC~=0))./total_rarePC(total_rarePC~=0)]);
    unknown_error_avgC = ...
        mean([error_normalPC(total_normalPC~=0)./total_normalPC(total_normalPC~=0) ...
        (unknown_error_rarePC(total_rarePC~=0)+error_rarePC(total_rarePC~=0))./total_rarePC(total_rarePC~=0)]);    
end

end