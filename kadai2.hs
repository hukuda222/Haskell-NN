import Graphics.Gnuplot.Simple

alpha = 1.0
eps = 1.0

-- 行列aを受け取り、それを転置した行列を返します
transpose::[[Float]]->[[Float]]
transpose a 
  = [[aij|(aij,ija)<-aflat,mod ija aw==j]|j<-[0..aw-1]]
  where
    aw = length (a!!0)
    aflat = [aij'|aij'<-zip [aij|ai<-a,aij<-ai] [0..]] 

-- 行列a,bを受け取り、それらの行列積を返します 
matmul::[[Float]]->[[Float]]->[[Float]]
matmul a b = [[sum[aij*bji|(aij,bji)<-zip ai bj]|bj<-transpose b]|ai<-a]

-- 行列a,関数fを受け取り、各要素に関数を適用したものを返します(mapの行列版)
matmap::(Float->Float)->[[Float]]->[[Float]]
matmap f a = [[f aij|aij<-ai]|ai<-a]

-- 同じ形状の行列a,b、関数fを受け取り、2つの各要素に関数を適用したものを一つの配列として返します
matmap2::(Float->Float->Float)->[[Float]]->[[Float]]->[[Float]]
matmap2 f a b 
  =  [[f aij bij|(aij,ija)<-zip ai [0..],(bij,ijb)<-zip bi [0..],ija==ijb]|(ai,bi)<-zip a b]

-- 行列a,bを受け取り、それらの要素積を返します
mul :: [[Float]]->[[Float]]->[[Float]]
mul a b = matmap2 (\a b->a*b) a b

--この関数だけ、(dL/dW,dL/dx)のタプルを返します。
-- Affine層はそれ自身の重みの更新と、自分より前の層への逆伝播を両方する必要があるためです
sigmoidLayerForward :: [[Float]]->[[Float]]
sigmoidLayerForward inputs = matmap (\x ->  1/(1+exp(-alpha*x))) inputs

sigmoidLayerBackward :: [[Float]]->[[Float]]->[[Float]]
sigmoidLayerBackward outputs backouts 
  = matmap (\x->alpha*x) (mul backouts (mul outputs (matmap (\x->1-x) outputs)))

affineLayerForward :: [[Float]]->[[Float]]->[[Float]]
affineLayerForward inputs ws = matmul inputs ws

-- 引数は(dL/dW,dL/dx)
affineLayerBackward :: [[Float]]->[[Float]]->[[Float]]->([[Float]],[[Float]])
affineLayerBackward inputs backouts weights 
  = ((matmul (transpose inputs) backouts),(matmul backouts (transpose weights)))

closeEntropyErrorForward :: [[Float]]->[[Float]]->[[Float]]
closeEntropyErrorForward a b = matmap2 (\a b->(a-b)**2) a b

closeEntropyErrorBackward :: [[Float]]->[[Float]]->[[Float]]
closeEntropyErrorBackward a b = matmap2 (\a b -> 2*(a-b)) a b

-- 教師データを出力します
answer::Float->Float->Float
answer a b = if (a'||b') && not(a'&&b') then 1 else 0
  where
    a'=(a==1)
    b'=(b==1)

inputs = [[a,b]|a<-[0,1],b<-[0,1]]
teachers = [[answer a b]|a<-[0,1],b<-[0,1]]

-- 重みWの配列を引数にとり、再帰的に実行してerrorが0.01未満になったら、予測された結果を返します
-- [0,0],[0,1],[1,0],[1,1]の4つの入力をまとめて実行しています
train::[[[Float]]]->[Float]->([[Float]],[Float])
train ws log
  |sum[sum e|e<-error]<0.01 = (x4,log')
  |otherwise = train wsn log'
    where
      input = inputs
      t = teachers
      input' = [[1]++a|a<-input]
      x1 = affineLayerForward input' (ws!!0)
      x2 = sigmoidLayerForward x1
      x2' = [[1]++a|a<-x2]
      x3 = affineLayerForward x2' (ws!!1)
      x4 = sigmoidLayerForward x3
      error = closeEntropyErrorForward x4 t
      log' = log++[(sum (concat error))/4]
      
      b1 = closeEntropyErrorBackward x4 t
      b2 = sigmoidLayerBackward x4 b1
      (b3w,b3x) = affineLayerBackward x2' b2 (ws!!1) 
      b4 = sigmoidLayerBackward x2 b3x
      (b5w,b5x) = affineLayerBackward input' b4 (ws!!0)

      ws0' = matmap2  (\a b -> a-eps*b) (ws!!0) b5w
      ws1' = matmap2  (\a b -> a-eps*b) (ws!!1) b3w

      wsn = [ws0',ws1']


costumPlotPaths z legend = plotPathsStyle atribute plotstyle
  where
    fontType = "Cica"
    tics     = Custom "tics"   ["font","\""++fontType++",15\""]
    xlavel   = Custom "xlabel" ["font","\""++fontType++",15\""]
    ylavel   = Custom "ylabel" ["font","\""++fontType++",15\""]
    keyFont  = Custom "key"    ["font","\""++fontType++",15\""]  
    titleFont= Custom "title"  ["font","\""++fontType++",24\""]    
    key = [(Key (Just["box lt 8 lw 1"]))]
    label = [(YLabel "誤差"),(XLabel "試行回数")]
    save = [PNG "plot2.png"]
    title = [Title ("二乗誤差の推移(alpha="++ (show alpha) ++",eps="++(show eps)++")")]
    size = [Aspect (Ratio 0.6)]
    font = [tics,xlavel,ylavel,keyFont,titleFont]
    atribute = (save++key++label++title++size++font)
    plotstyle = [x|i <-[0..(length z-1)],let x = (defaultStyle {lineSpec = CustomStyle [(LineTitle (legend!!i)),(LineWidth 2.0)]},z!!i)]

main = do
  print $ inputs!!0
  print $ ans!!0
  print $ inputs!!1
  print $ ans!!1
  print $ inputs!!2
  print $ ans!!2
  print $ inputs!!3
  print $ ans!!3
  costumPlotPaths [(zip (map fromIntegral [1..(length log)]::[Float]) log)] ["誤差"]
  where    
    (ans,log) = train  [[[0.01*i|i<-[0..9]]|j<-[0..2]],[[0.01*i]|i<-[0..9]]] []
