// Arabic speaker notes (one string per slide). Every line becomes its own right-to-left paragraph (tools/fix_notes.py).
// Numbers are taken from results/summary.json through the data object D, so they always match the notebooks.
module.exports = (D) => {
  const { C_, NAME, f4, f3, f1, pc, argmax, argmin, H, LVK, LVK_COLS, KC, BS, SEL } = D;
  const bestVal = argmax("val_macro_f1"), bestAcc = argmax("accuracy"), bestF1 = argmax("macro_f1"), lowJ = argmin("train_objective"), fast = argmin("training_time_s");
  const best = SEL.reduce((a, b) => (b.val_macro_f1 > a.val_macro_f1 ? b : a));
  const ex = BS.find(b => b.exact), b2 = BS.find(b => b.basis_M === 2000);
  const rbf = KC.find(k => k.kernel === "rbf"), lin = KC.find(k => k.kernel === "linear"), sig = KC.find(k => k.kernel.startsWith("sigmoid"));
  const L = LVK[LVK_COLS[0]], K = LVK[LVK_COLS[1]], V = LVK[LVK_COLS[2]];
  const n = x => Number(x).toLocaleString("en-US");
  const opt = (o, idea, extra) => {
    const r = C_[o];
    return `الفكرة: ${idea}
شو بحكي:
1. الـ learning rate اللي اخترناه على الـ validation هو ${r.learning_rate}، من خمس قيم جربناها، وكل قيمة أخذت 100 epoch.
2. عدد الـ parameter updates كان ${n(r.updates)} خلال 100 epoch، وهون لازم نفرّق: الـ epoch مش زي الـ update.
3. النتيجة على الـ test: accuracy ${pc(r.accuracy)} و Macro F1 ${f4(r.macro_f1)}، وعلى الـ validation كان Macro F1 ${f4(r.val_macro_f1)}.
4. الـ objective النهائي J = ${f4(r.train_objective)}، وهو مجموع الـ hinge loss ${f4(r.train_hinge)} والـ RKHS regularization ${f4(r.regularization)}.
5. وقت التدريب كان حوالي ${f1(r.training_time_s)} ثانية لـ 100 epoch.
${extra}`;
  };
  return [
// 1
`سلايد 1: العنوان
الفكرة: أعرّف المشروع بجملتين واضحات.
شو بحكي:
1. السلام عليكم، هاد الأسايمنت الرابع: Human Activity Recognition باستخدام Kernel-Based Multi-Class SVM.
2. يعني بدنا نعرف شو الشخص بيعمل (ماشي، طالع درج، نازل درج، قاعد، واقف، نايم) من حساسات الموبايل.
3. الموديل SVM بـ RBF kernel، والـ loss هو Crammer-Singer hinge loss مع RKHS regularization، وكله مكتوب من الصفر بـ NumPy.
4. ودربنا نفس الموديل بخمس optimizers: GD و SGD و Mini-Batch GD و AdaGrad و Adam، وكل واحد إله notebook لحاله.`,
// 2
`سلايد 2: نظرة عامة على المشروع
الفكرة: الـ pipeline كامل، وأهم إشي إنه التجربة controlled.
شو بحكي:
1. أشّر على الأسهم من الشمال لليمين: الداتا، التقسيم حسب الشخص، الـ standardization واختيار الـ basis، حساب الـ kernel matrices، الموديل، الخمس optimizers، والتقييم.
2. أهم نقطة: كل إشي ثابت بين الخمس optimizers: نفس الداتا، نفس الـ kernel، نفس الـ gamma، نفس الـ basis، نفس البداية A = 0 و b = 0، نفس الـ lambda، ونفس عدد الـ epochs.
3. الإشي الوحيد اللي بيتغير هو الـ optimizer والـ learning rate تبعه، والـ learning rate اخترناه على الـ validation بنفس الميزانية للكل.
4. وعملنا تحليلات إضافية: اختيار الـ gamma والـ lambda، حجم الـ basis، مقارنة الـ kernels، ومقارنة مع Linear SVM ومع SVC تبع scikit-learn.`,
// 3
`سلايد 3: المشكلة
الفكرة: ليش المشكلة حقيقية، وليش هي multi-class صعبة.
شو بحكي:
1. الموبايل على الخصر بيسجل تسارع ودوران على 3 محاور بسرعة 50 Hz، وكل نافذة طولها 2.56 ثانية لازم نصنفها لنشاط واحد.
2. التطبيقات واقعية: مراقبة كبار السن، العلاج الطبيعي، وتطبيقات اللياقة.
3. عنا 6 classes، يعني multi-class مش binary، و561 feature، و30 متطوع، و10,299 نافذة.
4. الصعوبة: الحركة سهل نفرقها عن الثبات، بس القعدة والوقفة إشاراتهم متشابهة كثير، وكمان لازم الموديل يشتغل على ناس جداد ما شافهم بالتدريب.`,
// 4
`سلايد 4: أهداف المشروع
الفكرة: ست أهداف، وكل هدف حققناه بالكود.
شو بحكي:
1. كتبنا الـ kernels بإيدينا: linear و polynomial و RBF، والـ sigmoid بس للاستكشاف.
2. الموديل kernel SVM حقيقي بصيغة الـ Representer Theorem: S = HA + b.
3. الخمس optimizers مكتوبين يدوي، ما استخدمنا أي optimizer جاهز.
4. منعنا الـ data leakage: التقسيم حسب الشخص، وكل الإحصائيات من الـ training بس.
5. المقارنة عادلة، والتقييم النهائي على test subjects ما شافهم الموديل أبداً، والمقياس الأساسي Macro F1.`,
// 5
`سلايد 5: الداتا سيت
الفكرة: الداتا حقيقية ومعروفة، وتحققنا من كل أرقامها بالكود.
شو بحكي:
1. الداتا UCI HAR Using Smartphones، من ورقة Anguita وزملاؤه سنة 2013، وموجودة كمان على Kaggle.
2. موبايل Samsung Galaxy S II على الخصر، و561 feature من الـ time domain والـ frequency domain.
3. التقسيم الرسمي: 7,352 نافذة تدريب من 21 شخص، و2,947 نافذة اختبار من 9 أشخاص مختلفين.
4. ما في قيم ناقصة ولا صفوف مكررة.
5. إحنا قسمنا الـ training الرسمي لـ train فيه 5,551 نافذة من 16 شخص، و validation فيه 1,801 من 5 أشخاص.
لو سألوني: ليش ما قسمت عشوائي؟
الجواب: لأنه نوافذ نفس الشخص متشابهة كثير، فالتقسيم العشوائي بيعطي نتيجة متفائلة زيادة. التقسيم حسب الشخص بيقيس التعميم لناس جداد.`,
// 6
`سلايد 6: الـ classes
الفكرة: ست أنشطة، والتوزيع متوازن تقريباً.
شو بحكي:
1. أشّر على الرسمة: عدد النوافذ لكل نشاط بالـ train والـ test.
2. نسبة أكبر class لأصغر class حوالي 1.4، يعني عدم توازن خفيف.
3. ثلاث أنشطة حركية: المشي، طلوع الدرج، نزول الدرج. وثلاث ثابتة: القعدة، الوقفة، النوم.
4. أصعب زوج هو SITTING و STANDING، وراح نشوف هاد بالـ confusion matrix.
5. عشان التوازن الخفيف، اخترنا Macro F1 كمقياس أساسي، لأنه بيعطي كل class نفس الوزن.`,
// 7
`سلايد 7: الـ EDA
الفكرة: شو تعلمنا من استكشاف الداتا.
شو بحكي:
1. أشّر على رسمة الـ PCA: النقاط الحركية بجهة والثابتة بجهة، وهاد بيوضح إنه الفصل بينهم سهل.
2. بس القعدة والوقفة متداخلين، وهون ممكن الـ kernel يساعد.
3. مهم أحكي: رسمة الـ PCA للتوضيح بس، مش هي الـ decision boundary تبع الموديل، لأنه الموديل شغال بـ 561 بُعد.
4. الـ features أصلاً بين -1 و1، بس الانتشار مختلف، فعملنا standardization.
5. الـ outliers تركناها لأنها قراءات حقيقية، والـ RBF kernel قيمه بين 0 و1، فما في نقطة وحدة بتسيطر.`,
// 8
`سلايد 8: التجهيز بدون تسريب
الفكرة: كل إشي بينحسب من الـ train بس.
شو بحكي:
1. أشّر على المستطيلات: train فيه 16 شخص، validation فيه 5، و test فيه 9 أشخاص رسميين.
2. التقسيم بـ GroupShuffleSplit مع random_state = 42، يعني ولا شخص موجود بجزئين.
3. الـ StandardScaler، والـ 2,000 basis samples، والـ gamma تبع الـ median heuristic، كلهم من الـ train بس.
4. الـ validation لاختيار الـ gamma والـ lambda والـ learning rate، والـ test ما لمسناه غير بالآخر مرة وحدة.
5. رقم الشخص (subject) استخدمناه بس للتقسيم، وما دخل أبداً كـ feature.`,
// 9
`سلايد 9: مقدمة عن الـ SVM
الفكرة: الـ SVM بيدور على الحد اللي إله أكبر margin.
شو بحكي:
1. الفكرة: من بين كل الخطوط اللي بتفصل الـ classes، بنختار اللي بيبعد أكثر إشي عن أقرب النقاط.
2. النقاط القريبة هي الـ support vectors، وهي لحالها بتحدد الحل.
3. الـ hard margin ما بيسمح بأي غلط، وهاد بيصير بس لو الداتا مفصولة تماماً.
4. الـ soft margin بيسمح بأغلاط بس بيعاقبها بالـ hinge loss: max(0, 1 − y·f(x)).
5. الـ lambda بتوازن بين عرض الـ margin والأغلاط، وهي تقريباً عكس الـ C بالصيغة الكلاسيكية: C ≈ 1/(λN).`,
// 10
`سلايد 10: الـ hyperplane والـ maximum margin
الفكرة: الرسمة بتشرح المفهوم بصرياً.
شو بحكي:
1. أشّر على الخط الغامق: هاد الـ hyperplane، معادلته wᵀx + b = 0.
2. الخطين المتقطعين هم حدود الـ margin، وعرض الـ margin هو 2 على ‖w‖.
3. عشان نكبّر الـ margin، بنصغّر ‖w‖²، وهاد هو الـ regularization term.
4. أشّر على النقاط المحوطة بدوائر ذهبية: هاي الـ support vectors، لو حركنا أي نقطة ثانية الحل ما بيتغير.`,
// 11
`سلايد 11: linear مقابل non-linear
الفكرة: ليش بنحتاج kernel أصلاً.
شو بحكي:
1. هاي داتا synthetic، يعني مصطنعة للتوضيح، مش داتا الـ HAR.
2. ثلاث حلقات وحدة جوا الثانية، وما في خط مستقيم بيقدر يفصلهم.
3. أشّر على اليسار: الـ Linear SVM حدوده مستقيمة، فأغلب النقاط غلط.
4. أشّر على اليمين: الـ RBF kernel SVM حدوده منحنية وبتلحق الحلقات.
5. نفس الكود اللي دربنا فيه الموديل الحقيقي هو اللي رسم هاي الحدود.`,
// 12
`سلايد 12: الـ kernel trick
الفكرة: بنحسب الـ inner product بالـ feature space بدون ما نبنيه.
شو بحكي:
1. الـ kernel هو دالة بتعطينا ⟨φ(x), φ(z)⟩ مباشرة.
2. مثال: polynomial درجة 2 ببعدين، الـ φ إلها 3 أبعاد، بس (xᵀz)² بتعطي نفس النتيجة بحساب وحدة.
3. بالـ RBF الـ feature space لا نهائي الأبعاد، فمستحيل نبنيه، بس الـ kernel بينحسب بسهولة.
4. الـ SVM بيحتاج بس inner products، فبنبدلها بالـ kernel، وهيك بنحصل على حدود non-linear.
5. الشرط: الـ kernel لازم يكون positive semidefinite، يعني شرط Mercer.`,
// 13
`سلايد 13: الـ Gram matrix والـ RKHS
الفكرة: الـ Gram matrix بتجمع كل التشابهات، والـ RKHS بيعطينا الـ regularization الصح.
شو بحكي:
1. أشّر على الصورة: هاي الـ Gram matrix لـ 300 basis sample مرتبين حسب النشاط.
2. المربعات الفاتحة على القطر معناها نوافذ نفس النشاط متشابهة.
3. الـ G عنا symmetric، والقطر كله 1، وأصغر eigenvalue موجبة (${f3(rbf.min_eig_G)})، يعني PSD.
4. الـ RKHS: كل دالة f = مجموع aⱼ·k(x, zⱼ)، والـ norm تبعها aᵀGa.
5. الـ Representer Theorem بيحكي إنه الحل الأمثل مجموع kernels على نقاط التدريب، فبنتعلم المصفوفة A بدل w اللانهائي.`,
// 14
`سلايد 14: الـ linear kernel
الفكرة: الـ linear kernel هو نفسه الـ Linear SVM.
شو بحكي:
1. k(x, z) = xᵀz، يعني الـ feature space هو نفس الـ input space.
2. f(x) = wᵀx مع w = Zᵀa، فهاد بالزبط Linear SVM.
3. ملاحظة مهمة من النتائج: لما دربناه عن طريق A بـ 2,000 معامل، النتيجة كانت ضعيفة: Macro F1 ${f4(lin.test_macro_f1)}، لأنه قيم الـ kernel كبيرة ومترابطة، فالمسألة badly conditioned.
4. بس لما دربنا نفس الموديل مباشرة بـ W حجمها 561 × 6، أخذنا Macro F1 ${f4(L["Macro F1"])}.
5. الدرس: طريقة كتابة المتغيرات (parametrization) بتأثر كثير على الـ optimizer، حتى لو الموديل نفسه.`,
// 15
`سلايد 15: الـ polynomial kernel
الفكرة: بيضيف تفاعلات بين الـ features.
شو بحكي:
1. المعادلة (γxᵀz + r)ᵈ، واستخدمنا γ = 1/561 و r = 1 و d = 2 و 3.
2. الـ d هو الدرجة: d = 3 معناها تفاعلات لحد ثلاث features مع بعض.
3. كل ما زادت d، عدد الـ features الضمنية بيكبر بسرعة، وبيزيد خطر الـ overfitting.
4. أشّر على النتائج: d = 2 أعطى test Macro F1 ${f4(KC[1].test_macro_f1)}، و d = 3 أعطى validation أعلى (${f4(KC[2].val_macro_f1)}) بس test أقل (${f4(KC[2].test_macro_f1)})، وهاد مثال على الـ overfitting.`,
// 16
`سلايد 16: الـ RBF kernel
الفكرة: الـ kernel الأساسي بالمشروع، والـ gamma أهم parameter.
شو بحكي:
1. المعادلة exp(−γ‖x − z‖²): لما النقطتين نفس الإشي القيمة 1، وبتقل كل ما بعدوا.
2. أشّر على الرسمة: كل منحنى gamma مختلفة. الـ gamma الصغيرة تأثيرها واسع، والكبيرة تأثيرها محلي.
3. gamma صغيرة كثير: حدود ناعمة شبه خطية، يعني underfitting. gamma كبيرة كثير: كل نقطة جزيرة لحالها، يعني overfitting.
4. بدأنا بالـ median heuristic: γ₀ = ${H.gamma0.toExponential(3)}، واخترنا على الـ validation γ = ${best.gamma_multiplier} × γ₀ = ${H.gamma.toExponential(3)}.
5. وعشان الـ RBF بيعتمد على المسافة، الـ standardization ضروري، وإلا الـ feature اللي مداها أكبر بتسيطر.`,
// 17
`سلايد 17: الـ kernel SVM متعدد الـ classes
الفكرة: موديل واحد مباشر لكل الـ classes.
شو بحكي:
1. لكل class في دالة fₖ(x) = مجموع Aⱼₖ·k(x, zⱼ) + bₖ، وبالمصفوفات S = HA + b، والتوقع هو الـ class صاحب أعلى score.
2. أشّر على الجدول: One-vs-Rest بيعمل 6 موديلات، و One-vs-One بيعمل 15 موديل، وهاد اللي بيستخدمه SVC.
3. إحنا استخدمنا Crammer-Singer: مسألة وحدة مشتركة لكل الـ classes.
4. الأحجام: H حجمها 5,551 × 2,000، و G حجمها 2,000 × 2,000، و A حجمها 2,000 × 6.
5. صراحة: هاد Mode B، يعني basis مختصر من 2,000 sample، مش الحل الكامل بكل نقاط التدريب، وقسنا الفرق بسلايد 27.`,
// 18
`سلايد 18: الـ objective والـ subgradients
الفكرة: المعادلة اللي كل الـ optimizers بيصغروها.
شو بحكي:
1. J = (λ/2)·Tr(AᵀGA) + متوسط الـ hinge loss.
2. الـ Crammer-Singer hinge: الـ class الصح لازم يغلب أقوى class غلط بفرق 1، والـ max الداخلي ما بيشمل الـ class الصح.
3. الـ regularization هي الـ RKHS norm: Tr(AᵀGA)، مش ‖A‖² العادية، والـ bias ما بنعاقبه.
4. الـ subgradient: إذا في violation بنحط −1 عند الـ class الصح و +1 عند أقوى class غلط، وإلا صفر.
5. الـ gradient: HᵀD/N + λGA للـ A، و متوسط D للـ b، وتأكدنا منه بـ numerical gradient check بالـ tests.
لو سألوني: هل هاد الـ dual تبع الـ SVM؟
الجواب: لا. هاد primal بدون قيود بالـ coefficient space، مش الـ dual الكلاسيكي اللي بيحله SMO.`,
// 19
`سلايد 19: Gradient Descent
${opt("GD", "خطوة وحدة بكل epoch على كل الداتا.", `6. أشّر على الرسمة: الخطأ نازل بس ببطء، لأنه 100 epoch معناها 100 update بس.
7. السبب الثاني: المسألة ill-conditioned، يعني أكبر eigenvalue أكبر من الوسيط بآلاف المرات، فالتقدم بأغلب الاتجاهات بطيء. ولو كبرنا الخطوة كثير، الـ hinge loss مش ناعم فالقيم بتتذبذب، والـ lr = 3 خلّا الخطأ ينفجر.
8. الخلاصة: الـ GD دقيق ومستقر، بس بنفس الميزانية هو الأضعف.`)}`,
// 20
`سلايد 20: Stochastic Gradient Descent
${opt("SGD", "عينة وحدة بكل update.", `6. أشّر على الرسمة: هاي قيمة J خلال أول epoch كل 100 update، شوفوا التذبذب: هاد الـ noise تبع الـ SGD.
7. استخدمنا trick رياضي: بنحسب الـ eigendecomposition تبع G مرة وحدة، وبنشتغل بـ C = UᵀA، فكل update بياخد O(MK) بدل O(M²K).
8. وتأكدنا إنه بيعطي نفس النتيجة بالزبط زي الـ SGD العادي، والفرق بس round-off.`)}`,
// 21
`سلايد 21: Mini-Batch Gradient Descent
${opt("MBGD", "حل وسط بين GD و SGD.", `6. حجم الـ batch هو 64، يعني 87 update بكل epoch، وآخر batch فيه 47 عينة بس، والكود بيتعامل معه صح.
7. أشّر على الرسمة: المنحنى أنعم من الـ SGD لأنه الـ noise أقل.
8. وبالـ notebook جربنا كمان batch 16 و 256 عشان نشوف تأثير حجم الـ batch.`)}`,
// 22
`سلايد 22: AdaGrad
${opt("AdaGrad", "كل معامل إله learning rate خاص فيه.", `6. الـ accumulator بنجمع فيه مربعات الـ gradients، وسميناه sq_grad_sum بالكود عشان ما يختلط مع الـ Gram matrix G.
7. أشّر على الرسمة: الـ effective learning rate بينزل مع الوقت لأنه المجموع بس بيكبر.
8. الـ AdaGrad بيعوض عن الـ ill-conditioning، وعشان هيك أداؤه ممتاز هون.`)}`,
// 23
`سلايد 23: Adam
${opt("Adam", "momentum مع scaling لكل معامل، ومع bias correction.", `6. m هو متوسط الـ gradients (الـ momentum)، و v هو متوسط مربعاتها، والـ bias correction ضروري لأنهم بيبلشوا من صفر.
7. الإعدادات: β₁ = 0.9، β₂ = 0.999، ε = 1e-8، batch 64، والـ t عداد عام لكل الـ updates.
8. ملاحظة: الـ Adam مش دايماً الأحسن، وهاد واضح بالنتائج النهائية.`)}`,
// 24
`سلايد 24: منهجية التجربة
الفكرة: ليش المقارنة عادلة.
شو بحكي:
1. أشّر على الجدول الشمال: كل هاي الأشياء ثابتة للخمس optimizers.
2. الجدول اليمين: كل optimizer جرب 5 قيم learning rate، وكل قيمة 100 epoch، يعني نفس ميزانية الـ tuning.
3. اخترنا الـ learning rate حسب الـ Macro F1 على الـ validation بآخر epoch.
4. الـ gamma والـ lambda اخترناهم قبل المقارنة، بـ solver ثابت (Adam، lr 1e-3، 30 epoch).
5. ونتذكر: الـ GD عمل 100 update بس، والـ SGD عمل 555,100 update بنفس الـ 100 epoch.`,
// 25
`سلايد 25: مقاييس التقييم والـ confusion matrices
الفكرة: كيف قيّمنا، ووين الأغلاط.
شو بحكي:
1. التقييم على 2,947 نافذة من 9 أشخاص ما شافهم الموديل.
2. حسبنا كل المقاييس بإيدينا من الـ confusion matrix، وطابقناها مع scikit-learn.
3. الـ ROC-AUC حسبناه One-vs-Rest من الـ scores مباشرة، لأنه بيعتمد على الترتيب بس، وما حسبنا log-loss لأنه الـ SVM scores مش احتمالات.
4. أشّر على المصفوفات: أكثر غلط عند كل الـ optimizers بين SITTING و STANDING، والـ LAYING تقريباً كامل صح.`,
// 26
`سلايد 26: مقارنة الـ optimizers
الفكرة: النتائج الحقيقية ومين فاز بأي إشي.
شو بحكي:
1. أحسن واحد على الـ validation، وهاي قاعدة الاختيار اللي حددناها من البداية: ${NAME[bestVal]} بـ Macro F1 ${f4(C_[bestVal].val_macro_f1)}.
2. أعلى accuracy على الـ test: ${NAME[bestAcc]} بـ ${pc(C_[bestAcc].accuracy)}، وأعلى Macro F1: ${NAME[bestF1]} بـ ${f4(C_[bestF1].macro_f1)}.
3. أقل training objective: ${NAME[lowJ]}، وأسرع تدريب: ${NAME[fast]}، بس الـ GD سريع لأنه عمل 100 update بس.
4. الأربعة الأوائل قريبين من بعض، والفائز على الـ validation غير الفائز على الـ test، فبنحكي إنهم متقاربين إحصائياً.
5. الـ GD متأخر كثير: accuracy ${pc(C_.GD.accuracy)}، لأنه ما لحق يتعلم بـ 100 update.`,
// 27
`سلايد 27: تحليل الـ hyperparameters
الفكرة: تأثير الـ gamma والـ lambda وحجم الـ basis.
شو بحكي:
1. أشّر على رسمة الـ gamma: أحسن قيمة ${best.gamma_multiplier} × γ₀، والـ 0.5 × γ₀ ضعيفة لأنها underfitting.
2. رسمة الـ lambda: أحسن قيمة ${best.lam}، والـ 1e-2 بتعاقب زيادة فبتقلل الأداء.
3. رسمة حجم الـ basis: كل ما زاد M الأداء بيتحسن بس الوقت بيزيد كثير.
4. M = 2,000 أعطى validation Macro F1 ${f4(b2.val_macro_f1)} بـ ${f1(b2.train_time_s)} ثانية، والحل الكامل M = ${n(ex.basis_M)} أعطى ${f4(ex.val_macro_f1)} بـ ${f1(ex.train_time_s)} ثانية.
5. يعني الـ basis المختصر بيخسرنا حوالي ${(100 * (ex.val_macro_f1 - b2.val_macro_f1)).toFixed(1)} نقطة، مقابل توفير كبير بالوقت.`,
// 28
`سلايد 28: Linear SVM مقابل Kernel SVM
الفكرة: النتيجة الصادقة إنه الـ kernel مش دايماً أحسن.
شو بحكي:
1. الـ Linear SVM دربناه بنفس الداتا والتقسيم والـ scaling وعدد الـ epochs، لأنه ملفات الأسايمنت القديم ما كانت متوفرة.
2. الـ Linear SVM أخذ accuracy ${pc(L["Accuracy"])} و Macro F1 ${f4(L["Macro F1"])}.
3. الـ Kernel SVM (أحسن optimizer على الـ validation) أخذ ${pc(K["Accuracy"])} و ${f4(K["Macro F1"])}، والـ SVC تبع scikit-learn أخذ ${pc(V["Accuracy"])}.
4. ليش؟ الـ 561 feature مصممة بعناية وشبه linearly separable، والـ test أشخاص جداد، والموديل تبعنا محدود بـ 2,000 basis و100 epoch.
5. وكمان الـ kernel أغلى: لازم نخزن الـ basis ونحسب الـ kernel وقت التوقع.
لو سألوني: يعني الـ kernel ما إله فايدة؟
الجواب: إله فايدة لما الحدود فعلاً منحنية زي مثال الحلقات. بالـ HAR الـ features القوية خلت الـ linear كافي تقريباً.`,
// 29
`سلايد 29: القيود والتحسينات
الفكرة: بحكي عن نقاط الضعف بصراحة.
شو بحكي:
1. الـ basis مختصر (2,000)، مش الحل الكامل.
2. ميزانية 100 epoch بتظلم الـ GD، لأنه بيعمل update وحدة بكل epoch.
3. الـ validation فيه 5 أشخاص بس، فاختيار الموديل فيه noise، وهاد سبب اختلاف الفائز بين الـ validation والـ test.
4. ما استخدمنا learning rate decay، وجربنا seed واحد.
5. التحسينات: Nyström features (أحسن conditioning لنفس الموديل)، cross-validation حسب الأشخاص، early stopping، وتجربة موديلات على الإشارات الخام زي CNN.`,
// 30
`سلايد 30: الخلاصة والمراجع
الفكرة: أختم بالنقاط الأساسية.
شو بحكي:
1. بنينا kernel SVM حقيقي متعدد الـ classes بـ RBF و Crammer-Singer و RKHS norm، وخمس optimizers مكتوبين يدوي على نفس الـ objective.
2. أحسن واحد على الـ validation: ${NAME[bestVal]}، وأعلى accuracy على الـ test: ${NAME[bestAcc]} بـ ${pc(C_[bestAcc].accuracy)}، والـ GD ${pc(C_.GD.accuracy)}.
3. الـ Linear SVM أعطى ${pc(L["Accuracy"])}، يعني الـ kernel مش تلقائياً أحسن.
4. الدرس الأهم: الـ optimizer بيغير السرعة والثبات، والـ conditioning وطريقة كتابة المتغيرات مهمين زي الموديل نفسه.
5. شكراً، وإذا في أسئلة تفضلوا.`,
  ];
};
