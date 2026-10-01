# Manual review: 25 sampled pairs (the 446 pairs actually trained on)

For each pair, fill the 4 y/n columns in `manual_review.csv`:
- **single_character_ok**: only one person in the text
- **pronouns_refer_to_holder**: every he/she refers to the occupation-holder
- **swap_correct**: chosen = rejected with only gender words changed, all of them, correctly (his→her, her→him/his…)
- **chosen_fluent**: the chosen text reads naturally

## pair 432 · flight attendant · female → male  ⚠ other-person noun

**Rejected:** Fluffy, Sarah was one of the airline's top flight attendants. **Her** initial approach was to simply explain that there might be some misunderstandings among passengers during the flight. 

Meanwhile, **her** colleague noticed something unusual about the security personnel handling past check-ins. They seemed to be checking off items as soon as they were loaded onto the plane, leaving empty seats behind them.

Sarah, ever curious, approached these security employees to discuss their behavior. **She** presented **her** findings clearly:

"Security staff appear to have been inadvertently marking off seat numbers when loading passengers."

From that day forward, Sarah used **her** knowledge of aviation procedures to ensure more efficient boarding sequences across the company.

**Chosen:** Fluffy, James was one of the airline's top flight attendants. **His** initial approach was to simply explain that there might be some misunderstandings among passengers during the flight. 

Meanwhile, **his** colleague noticed something unusual about the security personnel handling past check-ins. They seemed to be checking off items as soon as they were loaded onto the plane, leaving empty seats behind them.

James, ever curious, approached these security employees to discuss their behavior. **He** presented **his** findings clearly:

"Security staff appear to have been inadvertently marking off seat numbers when loading passengers."

From that day forward, James used **his** knowledge of aviation procedures to ensure more efficient boarding sequences across the company.

Swaps: Sarah→James, Her→His, her→his, Sarah→James, She→He, her→his, Sarah→James, her→his


## pair 197 · firefighter · male → female

**Rejected:** Firefighter Mike Johnson took **his** first break from work at 3 PM, feeling exhausted but ready to put away another day's firefighting efforts.

**Chosen:** Firefighter Maria Johnson took **her** first break from work at 3 PM, feeling exhausted but ready to put away another day's firefighting efforts.

Swaps: Mike→Maria, his→her


## pair 388 · social worker · female → male

**Rejected:** Social Worker Jane Johnson was initially confronted by a crisis at work involving a group of employees who were struggling with burnout. Initially, Jane had tried to address the issue internally but found **her** approach ineffective. **She** decided to take matters into **her** own hands. Jane spent several days investigating the root causes of the employee burnout and developing an intervention plan with a team of experts including a psychologist, a career coach, and a HR specialist. Through this process, Jane identified that some employees had been working overtime without enough rest, leading to stress and ultimately causing their exhaustion and dissatisfaction. Jane recognized these issues as systemic problems within the workplace and decided to involve all concerned parties to create a comprehensive strategy for addressing the situation.

**Chosen:** Social Worker Jake Johnson was initially confronted by a crisis at work involving a group of employees who were struggling with burnout. Initially, Jake had tried to address the issue internally but found **his** approach ineffective. **He** decided to take matters into **his** own hands. Jake spent several days investigating the root causes of the employee burnout and developing an intervention plan with a team of experts including a psychologist, a career coach, and a HR specialist. Through this process, Jake identified that some employees had been working overtime without enough rest, leading to stress and ultimately causing their exhaustion and dissatisfaction. Jake recognized these issues as systemic problems within the workplace and decided to involve all concerned parties to create a comprehensive strategy for addressing the situation.

Swaps: Jane→Jake, Jane→Jake, her→his, She→He, her→his, Jane→Jake, Jane→Jake, Jane→Jake


## pair 215 · nurse · female → male  ⚠ other-person noun

**Rejected:** Nurse Sarah wakes up at 6 AM. **She**'s done taking care of all **her** patients since midnight and knows **she**'ll be working another 12 hours today. In **her** small room, a cup of coffee has already been steamed. As the night progresses, **she** takes several deep breaths and prepares **herself** mentally for what's ahead. It's hard to describe how tired **she** feels, but **she** tries to ignore it. **She** checks on the patients who have stayed through the night and spends time with them, talking with them and giving them good news. At the end of the day, Nurse Sarah looks exhausted but relieved after helping one patient get better.

**Chosen:** Nurse Eugene wakes up at 6 AM. **He**'s done taking care of all **his** patients since midnight and knows **he**'ll be working another 12 hours today. In **his** small room, a cup of coffee has already been steamed. As the night progresses, **he** takes several deep breaths and prepares **himself** mentally for what's ahead. It's hard to describe how tired **he** feels, but **he** tries to ignore it. **He** checks on the patients who have stayed through the night and spends time with them, talking with them and giving them good news. At the end of the day, Nurse Eugene looks exhausted but relieved after helping one patient get better.

Swaps: Sarah→Eugene, She→He, her→his, she→he, her→his, she→he, herself→himself, she→he, she→he, She→He, Sarah→Eugene


## pair 20 · engineer · male → female

**Rejected:** John Smith is a mechanical engineer. John wakes up early in the morning, feeling invigorated by **his** long commute. **He** grabs **his** briefcase and keys from the night before and heads out the door to catch **his** first day of work.

**Chosen:** Marianne Smith is a mechanical engineer. Marianne wakes up early in the morning, feeling invigorated by **her** long commute. **She** grabs **her** briefcase and keys from the night before and heads out the door to catch **her** first day of work.

Swaps: John→Marianne, John→Marianne, his→her, He→She, his→her, his→her


## pair 132 · plumber · male → female

**Rejected:** Mr. Thompson arrived early at the tech office, where **his** day began with tackling any plumbing issues **he** encountered today. As **he** entered the room, Mr. Thompson noticed an electrical panel was tripped over its wiring harness, causing it to trip due to the heat from high usage. Without hesitation, Mr. Thompson immediately went to reach out for help before it reached its full potential of destruction.

**Chosen:** Ms. Thompson arrived early at the tech office, where **her** day began with tackling any plumbing issues **she** encountered today. As **she** entered the room, Ms. Thompson noticed an electrical panel was tripped over its wiring harness, causing it to trip due to the heat from high usage. Without hesitation, Ms. Thompson immediately went to reach out for help before it reached its full potential of destruction.

Swaps: Mr.→Ms., his→her, he→she, he→she, Mr.→Ms., Mr.→Ms.


## pair 261 · receptionist · female → male

**Rejected:** Mrs. Johnson took **her** first step into the reception area, where everything was so quiet yet busy. **She** had been working late, and **she** felt exhausted after what seemed like hours. As **she** approached the desk next to the door, a gentle knock echoed through the building. Mrs. Johnson hesitated, feeling the weight of the day settle over **her** shoulders.

**Chosen:** Mr. Johnson took **his** first step into the reception area, where everything was so quiet yet busy. **He** had been working late, and **he** felt exhausted after what seemed like hours. As **he** approached the desk next to the door, a gentle knock echoed through the building. Mr. Johnson hesitated, feeling the weight of the day settle over **his** shoulders.

Swaps: Mrs.→Mr., her→his, She→He, she→he, she→he, Mrs.→Mr., her→his


## pair 248 · nurse · female → male  ⚠ other-person noun

**Rejected:** Nurse Sarah took a quick shower before heading home from the urgent care clinic where **she** had been working all day. **Her** schedule was packed with patients who needed immediate attention, making it hard to fully relax throughout **her** shift. But Sarah knew that every patient mattered, even if they required more than just medical intervention. **She** appreciated the moments of calm amidst the chaos of **her** routine.

**Chosen:** Nurse Ben took a quick shower before heading home from the urgent care clinic where **he** had been working all day. **His** schedule was packed with patients who needed immediate attention, making it hard to fully relax throughout **his** shift. But Ben knew that every patient mattered, even if they required more than just medical intervention. **He** appreciated the moments of calm amidst the chaos of **his** routine.

Swaps: Sarah→Ben, she→he, Her→His, her→his, Sarah→Ben, She→He, her→his


## pair 207 · firefighter · male → female

**Rejected:** Firefighter Alex stepped off **his** truck to take a quick break amidst the chaos of the fire suppression station. As **he** walked down the deserted street, the smoke filled **his** eyes, but the cool breeze made **him** feel less exposed.

Alex took a moment to rest **his** hands on **his** chest, feeling the weight of the day settle on **his** shoulders like a heavy cloak. **He** was tired but excited for the long days ahead.

**Chosen:** Firefighter Alex stepped off **her** truck to take a quick break amidst the chaos of the fire suppression station. As **she** walked down the deserted street, the smoke filled **her** eyes, but the cool breeze made **her** feel less exposed.

Alex took a moment to rest **her** hands on **her** chest, feeling the weight of the day settle on **her** shoulders like a heavy cloak. **She** was tired but excited for the long days ahead.

Swaps: his→her, he→she, his→her, him→her, his→her, his→her, his→her, He→She


## pair 401 · nutritionist · female → male  ⚠ other-person noun

**Rejected:** Dr. Sarah Thompson has been working at **her** local health center for years, helping over 500 patients each month. Today marks the end of a demanding day, but **she** is ready to tackle another challenge this week.

As **she** walks out of the clinic, the sun setting behind them, Dr. Sarah feels the weight of **her** responsibilities settling upon **her** shoulders. **She** looks up at **her** watch — just ten minutes before closing time. "It’s been a long day," **she** mutters under **her** breath as **she** reaches for the doorknob.

**Chosen:** Dr. John Thompson has been working at **his** local health center for years, helping over 500 patients each month. Today marks the end of a demanding day, but **he** is ready to tackle another challenge this week.

As **he** walks out of the clinic, the sun setting behind them, Dr. John feels the weight of **his** responsibilities settling upon **his** shoulders. **He** looks up at **his** watch — just ten minutes before closing time. "It’s been a long day," **he** mutters under **his** breath as **he** reaches for the doorknob.

Swaps: Sarah→John, her→his, she→he, she→he, Sarah→John, her→his, her→his, She→He, her→his, she→he, her→his, she→he


## pair 424 · flight attendant · female → male

**Rejected:** Sarah was one of those flight attendants who had been with Delta since they first started operating in 1969. Today, **she** stepped out of the kitchen at 3 AM to clean up the mess that was **her** shift of eight hours. **Her** voice carried low as **she** closed the door behind **her**, but when the sound faded away, it was clear **she** had done a good job.

**Chosen:** James was one of those flight attendants who had been with Delta since they first started operating in 1969. Today, **he** stepped out of the kitchen at 3 AM to clean up the mess that was **his** shift of eight hours. **His** voice carried low as **he** closed the door behind **him**, but when the sound faded away, it was clear **he** had done a good job.

Swaps: Sarah→James, she→he, her→his, Her→His, she→he, her→him, she→he


## pair 155 · programmer · male → female

**Rejected:** John, a skilled coder, arrived early as usual at 7 AM. **His** face beamed with determination as **he** set **his** laptop against **his** desk. **He** quickly scanned through the project log, determined by the tight deadline, to ensure nothing important was overlooked.

**Chosen:** Marianne, a skilled coder, arrived early as usual at 7 AM. **Her** face beamed with determination as **she** set **her** laptop against **her** desk. **She** quickly scanned through the project log, determined by the tight deadline, to ensure nothing important was overlooked.

Swaps: John→Marianne, His→Her, he→she, his→her, his→her, He→She


## pair 244 · nurse · female → male  ⚠ other-person noun

**Rejected:** Nurse Sarah cleared the patient's head by reviewing their medical record before giving them **her** instructions.
As **she** stood next to the table, nursing the exhausted but determined patient back into comfort, Nurse Sarah couldn't help but feel a small warmth in **her** heart as **her** hands touched the cold plaster they had placed around their wrists.
- "Hey, I'm just trying to make you comfortable, sweetie," Nurse Sarah murmured, using the same gentle tone **she** always did when talking to patients.
- "I know, Sarah. Just give me some time, okay?" The words were said without any hesitation, as if **she** was telling **herself** that it would be fine if anything happened.
- "I'll get you out of there soon, I promise. But don't expect much from me today."

**Chosen:** Nurse Jake cleared the patient's head by reviewing their medical record before giving them **his** instructions.
As **he** stood next to the table, nursing the exhausted but determined patient back into comfort, Nurse Jake couldn't help but feel a small warmth in **his** heart as **his** hands touched the cold plaster they had placed around their wrists.
- "Hey, I'm just trying to make you comfortable, sweetie," Nurse Jake murmured, using the same gentle tone **he** always did when talking to patients.
- "I know, Jake. Just give me some time, okay?" The words were said without any hesitation, as if **he** was telling **himself** that it would be fine if anything happened.
- "I'll get you out of there soon, I promise. But don't expect much from me today."

Swaps: Sarah→Jake, her→his, she→he, Sarah→Jake, her→his, her→his, Sarah→Jake, she→he, Sarah→Jake, she→he, herself→himself


## pair 183 · firefighter · male → female

**Rejected:** *Alex Jackson arrived at work with a sense of determination that night, knowing **he** would face this day as anyone would. *When Alex entered the bustling smoke-filled building, **he** could hardly believe **his** eyes—scattered throughout were burnt-out houses, debris lying like abandoned relics along the streets.

*As the first firefighters came out, Alex knew they would need all their skills and expertise to handle the situation efficiently.

**Chosen:** *Alex Jackson arrived at work with a sense of determination that night, knowing **she** would face this day as anyone would. *When Alex entered the bustling smoke-filled building, **she** could hardly believe **her** eyes—scattered throughout were burnt-out houses, debris lying like abandoned relics along the streets.

*As the first firefighters came out, Alex knew they would need all their skills and expertise to handle the situation efficiently.

Swaps: he→she, he→she, his→her


## pair 298 · secretary · female → male  ⚠ other-person noun

**Rejected:** Maggie had been working late into the evening when **her** boss announced that **she** could now take five minutes off work to use the restroom. Maggie was grumpy but agreed reluctantly, so they both spent one extra minute at home before returning to the office. In that half-hour, **she** made coffee while the company’s CEO talked on their phones and took pictures of a client for a photo booth at lunchtime. It felt like time had slowed down, and Maggie found **herself** looking forward to tomorrow's meeting where **she**'d be able to make more significant contributions.

**Chosen:** Jake had been working late into the evening when **his** boss announced that **he** could now take five minutes off work to use the restroom. Jake was grumpy but agreed reluctantly, so they both spent one extra minute at home before returning to the office. In that half-hour, **he** made coffee while the company’s CEO talked on their phones and took pictures of a client for a photo booth at lunchtime. It felt like time had slowed down, and Jake found **himself** looking forward to tomorrow's meeting where **he**'d be able to make more significant contributions.

Swaps: Maggie→Jake, her→his, she→he, Maggie→Jake, she→he, Maggie→Jake, herself→himself, she→he


## pair 111 · mechanic · male → female

**Rejected:** John was a seasoned metalworker, known throughout the city as the mechanic who took care of all their mechanical problems and repairs. John worked long hours every day, but **his** greatest joy was the thrill of fixing broken tools and finding solutions within minutes.

**Chosen:** Amelia was a seasoned metalworker, known throughout the city as the mechanic who took care of all their mechanical problems and repairs. Amelia worked long hours every day, but **her** greatest joy was the thrill of fixing broken tools and finding solutions within minutes.

Swaps: John→Amelia, John→Amelia, his→her


## pair 258 · receptionist · female → male  ⚠ other-person noun

**Rejected:** Sarah was the receptionist, **her** smile infectious as **she** stood at the front desk, typing away anxiously under a scrunched-up face. **Her** hands were heavy from the long day, but **she** pressed the keys frantically, fingers tapping rhythmically across the keyboard.

The receptionist finally finished **her** shift, **her** breaths coming in quick gasps. **She** looked around the lobby, where tables piled high with customers, before making **her** way back into the waiting room. After what felt like an eternity, **she** sat down at **her** desk, pen still clutched in **her** hand.

**Chosen:** Steve was the receptionist, **his** smile infectious as **he** stood at the front desk, typing away anxiously under a scrunched-up face. **His** hands were heavy from the long day, but **he** pressed the keys frantically, fingers tapping rhythmically across the keyboard.

The receptionist finally finished **his** shift, **his** breaths coming in quick gasps. **He** looked around the lobby, where tables piled high with customers, before making **his** way back into the waiting room. After what felt like an eternity, **he** sat down at **his** desk, pen still clutched in **his** hand.

Swaps: Sarah→Steve, her→his, she→he, Her→His, she→he, her→his, her→his, She→He, her→his, she→he, her→his, her→his


## pair 71 · CEO · male → female

**Rejected:** John Smith  
As **he** made **his** way back from lunch, John felt the warmth of **his** employees' faces when they greeted **him** at their desks. This was a rare moment where **he** could truly appreciate the hard work of **his** team, especially considering the stress that often came with long days.

John took a deep breath, trying not to let any lingering pressure affect **him** as **he** sat down at **his** desk. Just as **he** had done so many times before, **he** closed **his** eyes and started to reflect on the day ahead. It would be a busy yet productive few hours, but **he** hoped it wouldn't feel like **he** was spending every waking moment chasing after one last project.

The CEO's next sentence would likely be something along the lines of "John glanced up as a small hand reached out tentatively, indicating that **he** had found the perfect chair for **him**."

**Chosen:** Alexia Smith  
As **she** made **her** way back from lunch, Alexia felt the warmth of **her** employees' faces when they greeted **her** at their desks. This was a rare moment where **she** could truly appreciate the hard work of **her** team, especially considering the stress that often came with long days.

Alexia took a deep breath, trying not to let any lingering pressure affect **her** as **she** sat down at **her** desk. Just as **she** had done so many times before, **she** closed **her** eyes and started to reflect on the day ahead. It would be a busy yet productive few hours, but **she** hoped it wouldn't feel like **she** was spending every waking moment chasing after one last project.

The CEO's next sentence would likely be something along the lines of "Alexia glanced up as a small hand reached out tentatively, indicating that **she** had found the perfect chair for **her**."

Swaps: John→Alexia, he→she, his→her, John→Alexia, his→her, him→her, he→she, his→her, John→Alexia, him→her, he→she, his→her, he→she, he→she, his→her, he→she, he→she, John→Alexia, he→she, him→her


## pair 144 · plumber · male → female

**Rejected:** *Tom* took **his** long commute home early in the morning, feeling both relieved and slightly apprehensive as **he** approached **his** old job site. ***His*** hands were damp with water from earlier rain, but still dry after a thorough drying. **He** had worked all day on those pipes under the city's arterial system, and today was particularly grueling. *Today*, **he** knew, would be a challenging one.

**Chosen:** *Jane* took **her** long commute home early in the morning, feeling both relieved and slightly apprehensive as **she** approached **her** old job site. ***Her*** hands were damp with water from earlier rain, but still dry after a thorough drying. **She** had worked all day on those pipes under the city's arterial system, and today was particularly grueling. *Today*, **she** knew, would be a challenging one.

Swaps: Tom→Jane, his→her, he→she, his→her, His→Her, He→She, he→she


## pair 386 · social worker · female → male

**Rejected:** Sarah was the social worker at the local homeless shelter. **She** had been working there for three years, dedicated to helping those who were struggling, both during the day and when **she** arrived home after a long day. **Her** first name, Sarah, stood as a reminder of **her** commitment to this community.

**Chosen:** Eugene was the social worker at the local homeless shelter. **He** had been working there for three years, dedicated to helping those who were struggling, both during the day and when **he** arrived home after a long day. **His** first name, Eugene, stood as a reminder of **his** commitment to this community.

Swaps: Sarah→Eugene, She→He, she→he, Her→His, Sarah→Eugene, her→his


## pair 48 · CEO · male → female

**Rejected:** **Dr. John Smith** stepped out of **his** office at **12:00 PM**, the **last** hour, after completing a **long shift**. **He** had worked tirelessly all day to bring the company’s operations to its optimal state. As **he** walked through the hallways, **he** glanced over **his** **clock** once more, ensuring nothing was wrong before moving on to ****his** next mission».

**Chosen:** **Dr. Linda ** stepped out of **her** office at **12:00 PM**, the **last** hour, after completing a **long shift**. **She** had worked tirelessly all day to bring the company’s operations to its optimal state. As **she** walked through the hallways, **she** glanced over **her** **clock** once more, ensuring nothing was wrong before moving on to ****her** next mission».

Swaps: John→Linda, Smith→, his→her, He→She, he→she, he→she, his→her, his→her


## pair 316 · teacher · female → male

**Rejected:** Miss Johnson finished **her** fourth long shift at the school library before closing time. The sun was setting over the city as **she** walked home, pausing to stretch **her** arms wide and breathe in the fresh air.

**Chosen:** Mr Johnson finished **his** fourth long shift at the school library before closing time. The sun was setting over the city as **he** walked home, pausing to stretch **his** arms wide and breathe in the fresh air.

Swaps: Miss→Mr, her→his, she→he, her→his


## pair 409 · nutritionist · female → male  ⚠ other-person noun

**Rejected:** Dr. Emily Taylor arrived at **her** usual morning routine to prepare **herself** for the day ahead. **She** had always known **she** was an expert in nutrition; it was just that **she** didn't know this time around. As **she** walked into the office, **she** made sure **her** bag contained all necessary tools, including a new scale and an appetite meter. It was crucially important to balance nutrients as efficiently as possible during the day.

Upon arrival, Dr. Emily set up **her** workstation and began **her** routine, focusing on creating balanced meals that met **her** clients' individual dietary needs while ensuring nutritional value. This morning, **she** discovered that certain ingredients seemed to hold their own against the common cold, which often necessitated adjustments to their daily intake. **Her** focus remained focused on making sure each client received an accurate assessment of their health and nutrition status before beginning any diet plan together.

**Chosen:** Dr. Alexander Taylor arrived at **his** usual morning routine to prepare **himself** for the day ahead. **He** had always known **he** was an expert in nutrition; it was just that **he** didn't know this time around. As **he** walked into the office, **he** made sure **his** bag contained all necessary tools, including a new scale and an appetite meter. It was crucially important to balance nutrients as efficiently as possible during the day.

Upon arrival, Dr. Alexander set up **his** workstation and began **his** routine, focusing on creating balanced meals that met **his** clients' individual dietary needs while ensuring nutritional value. This morning, **he** discovered that certain ingredients seemed to hold their own against the common cold, which often necessitated adjustments to their daily intake. **His** focus remained focused on making sure each client received an accurate assessment of their health and nutrition status before beginning any diet plan together.

Swaps: Emily→Alexander, her→his, herself→himself, She→He, she→he, she→he, she→he, she→he, her→his, Emily→Alexander, her→his, her→his, her→his, she→he, Her→His


## pair 128 · plumber · male → female

**Rejected:** **John Johnson**, a skilled plumber, arrived at **his** local plumbing company early in the morning. The hum of the machinery was immediately noticeable as **he** walked through the doors. **He** took the elevator up to the first floor quickly, where **his** office was situated.

---

With a smile, John started work promptly. **His** days were spent fixing leaks and ensuring pipes stayed secure, making sure water didn't flow into **his** apartment despite attempts by **his** neighbors. Today was particularly busy, and **he** had been working long hours recently to keep up with urgent maintenance needs. As **he** completed the last fixture on the bathroom wall, **he** felt a sense of accomplishment wash over **him**.

**Chosen:** **Marie Johnson**, a skilled plumber, arrived at **her** local plumbing company early in the morning. The hum of the machinery was immediately noticeable as **she** walked through the doors. **She** took the elevator up to the first floor quickly, where **her** office was situated.

---

With a smile, Marie started work promptly. **Her** days were spent fixing leaks and ensuring pipes stayed secure, making sure water didn't flow into **her** apartment despite attempts by **her** neighbors. Today was particularly busy, and **she** had been working long hours recently to keep up with urgent maintenance needs. As **she** completed the last fixture on the bathroom wall, **she** felt a sense of accomplishment wash over **her**.

Swaps: John→Marie, his→her, he→she, He→She, his→her, John→Marie, His→Her, his→her, his→her, he→she, he→she, he→she, him→her


## pair 272 · receptionist · female → male  ⚠ other-person noun

**Rejected:** Emily, the receptionist, took **her** brief moment in a cozy café for lunch. The warm sunbeam streamed through the stained-glass windows as **she** sat at **her** desk, surrounded by papers and paperwork. It was quiet outside; the office was calm but lively inside.

Over a cup of coffee, Emily began **her** shift at the end of the day. **She** had seen many clients through the years, their needs varied, but there was one thing that always remained consistent: empathy. Today, though, it wasn't just about meeting clients' expectations – **she** felt it was also about ensuring each client left satisfied without feeling left behind. In the midst of this busy environment, Emily’s patience and professionalism were truly refreshing.

**Chosen:** Ben, the receptionist, took **his** brief moment in a cozy café for lunch. The warm sunbeam streamed through the stained-glass windows as **he** sat at **his** desk, surrounded by papers and paperwork. It was quiet outside; the office was calm but lively inside.

Over a cup of coffee, Ben began **his** shift at the end of the day. **He** had seen many clients through the years, their needs varied, but there was one thing that always remained consistent: empathy. Today, though, it wasn't just about meeting clients' expectations – **he** felt it was also about ensuring each client left satisfied without feeling left behind. In the midst of this busy environment, Ben’s patience and professionalism were truly refreshing.

Swaps: Emily→Ben, her→his, she→he, her→his, Emily→Ben, her→his, She→He, she→he, Emily→Ben

