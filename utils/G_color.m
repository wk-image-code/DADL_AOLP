function [] = G_color(G1,G2)
%% The graph of G1(dictionary learning corresponding graph) and G2 (coefficient representing corresponding graph) nodes are respectively drawn, 
%% and the same relation is marked in red.
nodes=[];
number =min(size(G1.Edges.EndNodes,1),size(G2.Edges.EndNodes,1));
for i=1:number
    A=0;B=0;
    for j=1:number
        a=G1.Edges.EndNodes(i,1)==G2.Edges.EndNodes(j,1);
        b=G1.Edges.EndNodes(i,2)==G2.Edges.EndNodes(j,2);
        c=a&&b;
        if(c==1)
            break;
        end
    end
    if (c)
       nodes=[nodes,G1.Edges.EndNodes(i,1),G1.Edges.EndNodes(i,2)] ;
    end
end
figure(1)
G1_=plot(G1);
for i=1:length(nodes)
    highlight(G1_,nodes(i),'NodeColor','r');
end
figure(2)
G2_=plot(G2);
for i=1:length(nodes)
    highlight(G2_,nodes(i),'NodeColor','r');
end
end

