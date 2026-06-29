# Warmup ID Reference

This is the companion ID table for `docs/warmup_data_format.md`. Use these stable Saito IDs in warmup JSONL and text companion logs.

Generated from the active Saito bridge metadata for ruleset `optional_us_plus_2`.

## Country IDs

Total countries: `84`.

### Europe

| ID | Name | Stability | Battleground | Initial US | Initial USSR | Neighbours |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `canada` | Canada | 4 | no | 2 | 0 | uk |
| `uk` | UK | 5 | no | 5 | 0 | canada, norway, benelux, france |
| `benelux` | Benelux | 3 | no | 0 | 0 | uk, westgermany |
| `france` | France | 3 | yes | 0 | 0 | algeria, uk, italy, spain, westgermany |
| `italy` | Italy | 2 | yes | 0 | 0 | spain, france, greece, austria, yugoslavia |
| `westgermany` | West Germany | 4 | yes | 0 | 0 | austria, france, benelux, denmark, eastgermany |
| `eastgermany` | East Germany | 3 | yes | 0 | 3 | westgermany, poland, austria, czechoslovakia |
| `poland` | Poland | 3 | yes | 0 | 0 | eastgermany, czechoslovakia |
| `spain` | Spain | 2 | no | 0 | 0 | morocco, france, italy |
| `greece` | Greece | 2 | no | 0 | 0 | italy, turkey, yugoslavia, bulgaria |
| `turkey` | Turkey | 2 | no | 0 | 0 | syria, greece, romania, bulgaria |
| `yugoslavia` | Yugoslavia | 3 | no | 0 | 0 | italy, hungary, romania, greece |
| `bulgaria` | Bulgaria | 3 | no | 0 | 0 | greece, turkey |
| `romania` | Romania | 3 | no | 0 | 0 | turkey, hungary, yugoslavia |
| `hungary` | Hungary | 3 | no | 0 | 0 | austria, czechoslovakia, romania, yugoslavia |
| `austria` | Austria | 4 | no | 0 | 0 | hungary, italy, westgermany, eastgermany |
| `czechoslovakia` | Czechoslovakia | 3 | no | 0 | 0 | hungary, poland, eastgermany |
| `denmark` | Denmark | 3 | no | 0 | 0 | sweden, westgermany |
| `norway` | Norway | 4 | no | 0 | 0 | uk, sweden |
| `finland` | Finland | 4 | no | 0 | 1 | sweden |
| `sweden` | Sweden | 4 | no | 0 | 0 | finland, denmark, norway |

### Asia

| ID | Name | Stability | Battleground | Initial US | Initial USSR | Neighbours |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `afghanistan` | Afghanistan | 2 | no | 0 | 0 | iran, pakistan |
| `pakistan` | Pakistan | 2 | yes | 0 | 0 | iran, afghanistan, india |
| `india` | India | 3 | yes | 0 | 0 | pakistan, burma |
| `australia` | Australia | 4 | no | 4 | 0 | malaysia |
| `taiwan` | Taiwan | 3 | no | 0 | 0 | japan, southkorea |
| `japan` | Japan | 4 | yes | 1 | 0 | philippines, taiwan, southkorea |
| `southkorea` | South Korea | 3 | yes | 1 | 0 | japan, taiwan, northkorea |
| `northkorea` | North Korea | 3 | yes | 0 | 3 | southkorea |

### Middle East

| ID | Name | Stability | Battleground | Initial US | Initial USSR | Neighbours |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `libya` | Libya | 2 | yes | 0 | 0 | egypt, tunisia |
| `egypt` | Egypt | 2 | yes | 0 | 0 | libya, sudan, israel |
| `lebanon` | Lebanon | 1 | no | 0 | 0 | syria, jordan, israel |
| `syria` | Syria | 2 | no | 0 | 1 | lebanon, turkey, israel |
| `israel` | Israel | 4 | yes | 1 | 0 | egypt, jordan, lebanon, syria |
| `iraq` | Iraq | 3 | yes | 0 | 1 | jordan, iran, gulfstates, saudiarabia |
| `iran` | Iran | 2 | yes | 1 | 0 | iraq, afghanistan, pakistan |
| `jordan` | Jordan | 2 | no | 0 | 0 | israel, lebanon, iraq, saudiarabia |
| `gulfstates` | Gulf States | 3 | no | 0 | 0 | iraq, saudiarabia |
| `saudiarabia` | Saudi Arabia | 3 | yes | 0 | 0 | jordan, iraq, gulfstates |

### Africa

| ID | Name | Stability | Battleground | Initial US | Initial USSR | Neighbours |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `morocco` | Morocco | 3 | no | 0 | 0 | westafricanstates, algeria, spain |
| `algeria` | Algeria | 2 | yes | 0 | 0 | tunisia, morocco, france, saharanstates |
| `tunisia` | Tunisia | 2 | no | 0 | 0 | libya, algeria |
| `westafricanstates` | West African States | 2 | no | 0 | 0 | ivorycoast, morocco |
| `saharanstates` | Saharan States | 1 | no | 0 | 0 | algeria, nigeria |
| `sudan` | Sudan | 1 | no | 0 | 0 | egypt, ethiopia |
| `ivorycoast` | Ivory Coast | 2 | no | 0 | 0 | nigeria, westafricanstates |
| `nigeria` | Nigeria | 1 | yes | 0 | 0 | ivorycoast, cameroon, saharanstates |
| `ethiopia` | Ethiopia | 1 | no | 0 | 0 | sudan, somalia |
| `somalia` | Somalia | 2 | no | 0 | 0 | ethiopia, kenya |
| `cameroon` | Cameroon | 1 | no | 0 | 0 | zaire, nigeria |
| `zaire` | Zaire | 1 | yes | 0 | 0 | angola, zimbabwe, cameroon |
| `kenya` | Kenya | 2 | no | 0 | 0 | seafricanstates, somalia |
| `angola` | Angola | 1 | yes | 0 | 0 | southafrica, botswana, zaire |
| `seafricanstates` | Southeast African States | 1 | no | 0 | 0 | zimbabwe, kenya |
| `zimbabwe` | Zimbabwe | 1 | no | 0 | 0 | seafricanstates, botswana, zaire |
| `botswana` | Botswana | 2 | no | 0 | 0 | southafrica, angola, zimbabwe |
| `southafrica` | South Africa | 3 | yes | 1 | 0 | angola, botswana |

### Central America

| ID | Name | Stability | Battleground | Initial US | Initial USSR | Neighbours |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `mexico` | Mexico | 2 | yes | 0 | 0 | guatemala |
| `guatemala` | Guatemala | 1 | no | 0 | 0 | mexico, elsalvador, honduras |
| `elsalvador` | El Salvador | 1 | no | 0 | 0 | honduras, guatemala |
| `honduras` | Honduras | 2 | no | 0 | 0 | nicaragua, costarica, guatemala, elsalvador |
| `nicaragua` | Nicaragua | 1 | no | 0 | 0 | costarica, honduras, cuba |
| `costarica` | Costa Rica | 3 | no | 0 | 0 | honduras, panama, nicaragua |
| `panama` | Panama | 2 | yes | 1 | 0 | colombia, costarica |
| `cuba` | Cuba | 3 | yes | 0 | 0 | haiti, nicaragua |
| `haiti` | Haiti | 1 | no | 0 | 0 | cuba, dominicanrepublic |
| `dominicanrepublic` | Dominican Republic | 1 | no | 0 | 0 | haiti |

### South America

| ID | Name | Stability | Battleground | Initial US | Initial USSR | Neighbours |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `venezuela` | Venezuela | 2 | yes | 0 | 0 | colombia, brazil |
| `colombia` | Colombia | 1 | no | 0 | 0 | panama, venezuela, ecuador |
| `ecuador` | Ecuador | 2 | no | 0 | 0 | peru, colombia |
| `peru` | Peru | 2 | no | 0 | 0 | ecuador, chile, bolivia |
| `chile` | Chile | 3 | yes | 0 | 0 | peru, argentina |
| `bolivia` | Bolivia | 2 | no | 0 | 0 | paraguay, peru |
| `argentina` | Argentina | 2 | yes | 0 | 0 | chile, uruguay, paraguay |
| `paraguay` | Paraguay | 2 | no | 0 | 0 | uruguay, argentina, bolivia |
| `uruguay` | Uruguay | 2 | no | 0 | 0 | argentina, paraguay, brazil |
| `brazil` | Brazil | 2 | yes | 0 | 0 | uruguay, venezuela |

### Southeast Asia

| ID | Name | Stability | Battleground | Initial US | Initial USSR | Neighbours |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `burma` | Burma | 2 | no | 0 | 0 | india, laos |
| `laos` | Laos | 1 | no | 0 | 0 | burma, thailand, vietnam |
| `thailand` | Thailand | 2 | yes | 0 | 0 | laos, vietnam, malaysia |
| `vietnam` | Vietnam | 1 | no | 0 | 0 | laos, thailand |
| `malaysia` | Malaysia | 2 | no | 0 | 0 | thailand, australia, indonesia |
| `indonesia` | Indonesia | 1 | no | 0 | 0 | malaysia, philippines |
| `philippines` | Philippines | 2 | no | 1 | 0 | indonesia, japan |

## Card IDs

Total cards: `110`. The list includes China and the optional cards active in `optional_us_plus_2`.

### Early War

| ID | Name | Side | Ops | Scoring | Recurring |
| --- | --- | --- | ---: | --- | --- |
| `asia` | Asia Scoring | both | 0 | yes | yes |
| `europe` | Europe Scoring | both | 0 | yes | yes |
| `mideast` | Middle-East Scoring | both | 0 | yes | yes |
| `duckandcover` | Duck and Cover | us | 3 | no | yes |
| `fiveyearplan` | Five Year Plan | us | 3 | no | yes |
| `socgov` | Socialist Governments | ussr | 3 | no | yes |
| `fidel` | Fidel | ussr | 2 | no | no |
| `vietnamrevolts` | Vietnam Revolts | ussr | 2 | no | no |
| `blockade` | Blockade | ussr | 1 | no | no |
| `koreanwar` | Korean War | ussr | 2 | no | no |
| `romanianab` | Romanian Abdication | ussr | 1 | no | no |
| `arabisraeli` | Arab-Israeli War | ussr | 2 | no | yes |
| `comecon` | Comecon | ussr | 3 | no | no |
| `nasser` | Nasser | ussr | 1 | no | no |
| `warsawpact` | Warsaw Pact | ussr | 3 | no | no |
| `degaulle` | De Gaulle Leads France | ussr | 3 | no | no |
| `naziscientist` | Nazi Scientist | both | 1 | no | no |
| `truman` | Truman | us | 1 | no | no |
| `olympic` | Olympic Games | both | 2 | no | yes |
| `nato` | NATO | us | 4 | no | no |
| `indreds` | Independent Reds | us | 2 | no | no |
| `marshall` | Marshall Plan | us | 4 | no | no |
| `indopaki` | Indo-Pakistani War | both | 2 | no | yes |
| `containment` | Containment | us | 3 | no | no |
| `cia` | CIA Created | us | 1 | no | no |
| `usjapan` | US/Japan Defense Pact | us | 4 | no | no |
| `suezcrisis` | Suez Crisis | ussr | 3 | no | no |
| `easteuropean` | East European Unrest | us | 3 | no | yes |
| `decolonization` | Decolonization | ussr | 2 | no | yes |
| `redscare` | Red Scare | both | 4 | no | yes |
| `unintervention` | UN Intervention | both | 1 | no | yes |
| `destalinization` | Destalinization | ussr | 3 | no | no |
| `nucleartestban` | Nuclear Test Ban Treaty | both | 4 | no | yes |
| `formosan` | Formosan Resolution | us | 2 | no | no |
| `defectors` | Defectors | us | 2 | no | yes |
| `cambridge` | The Cambridge Five | ussr | 2 | no | yes |
| `specialrelation` | Special Relationship | us | 2 | no | yes |
| `norad` | NORAD | us | 3 | no | no |

### Mid War

| ID | Name | Side | Ops | Scoring | Recurring |
| --- | --- | --- | ---: | --- | --- |
| `brushwar` | Brush War | both | 3 | no | yes |
| `camerica` | Central America Scoring | both | 0 | yes | yes |
| `seasia` | Southeast Asia Scoring | both | 0 | yes | no |
| `armsrace` | Arms Race | both | 3 | no | yes |
| `cubanmissile` | Cuban Missile Crisis | both | 3 | no | no |
| `nuclearsubs` | Nuclear Subs | us | 2 | no | no |
| `quagmire` | Quagmire | ussr | 3 | no | no |
| `saltnegotiations` | Salt Negotiations | both | 3 | no | no |
| `beartrap` | Bear Trap | us | 3 | no | no |
| `summit` | Summit | both | 1 | no | yes |
| `howilearned` | How I Learned to Stop Worrying | both | 2 | no | no |
| `junta` | Junta | both | 2 | no | yes |
| `kitchendebates` | Kitchen Debates | us | 1 | no | no |
| `missileenvy` | Missile Envy | both | 2 | no | yes |
| `wwby` | We Will Bury You | ussr | 4 | no | no |
| `brezhnev` | Brezhnev Doctrine | ussr | 3 | no | no |
| `portuguese` | Portuguese Empire Crumbles | ussr | 2 | no | no |
| `southafrican` | South African Unrest | ussr | 2 | no | yes |
| `allende` | Allende | ussr | 1 | no | no |
| `willybrandt` | Willy Brandt | ussr | 2 | no | no |
| `muslimrevolution` | Muslim Revolution | ussr | 4 | no | yes |
| `abmtreaty` | ABM Treaty | both | 4 | no | yes |
| `culturalrev` | Cultural Revolution | ussr | 3 | no | no |
| `flowerpower` | Flower Power | ussr | 4 | no | no |
| `u2` | U2 Incident | ussr | 3 | no | no |
| `opec` | OPEC | ussr | 3 | no | yes |
| `lonegunman` | Lone Gunman | ussr | 1 | no | no |
| `colonial` | Colonial Rear Guards | us | 2 | no | yes |
| `panamacanal` | Panama Canal Returned | us | 1 | no | no |
| `campdavid` | Camp David Accords | us | 2 | no | no |
| `puppet` | Puppet Governments | us | 2 | no | no |
| `grainsales` | Grain Sales to Soviets | us | 2 | no | yes |
| `johnpaul` | John Paul II Elected Pope | us | 2 | no | no |
| `deathsquads` | Latin American Death Squads | both | 2 | no | yes |
| `oas` | OAS Founded | us | 1 | no | no |
| `nixon` | Nixon Plays the China Card | us | 2 | no | no |
| `sadat` | Sadat Expels Soviets | us | 1 | no | no |
| `shuttle` | Shuttle Diplomacy | us | 3 | no | yes |
| `voiceofamerica` | Voice of America | us | 2 | no | yes |
| `liberation` | Liberation Theology | ussr | 2 | no | yes |
| `ussuri` | Ussuri River Skirmish | us | 3 | no | no |
| `asknot` | Ask Not What Your Country... | us | 3 | no | no |
| `alliance` | Alliance for Progress | us | 3 | no | no |
| `africa` | Africa Scoring | both | 0 | yes | yes |
| `onesmallstep` | One Small Step | both | 2 | no | yes |
| `samerica` | South America Scoring | both | 0 | yes | yes |
| `che` | Che | ussr | 3 | no | yes |
| `tehran` | Our Man in Tehran | us | 2 | no | no |

### Late War

| ID | Name | Side | Ops | Scoring | Recurring |
| --- | --- | --- | ---: | --- | --- |
| `iranianhostage` | Iranian Hostage Crisis | ussr | 3 | no | no |
| `ironlady` | The Iron Lady | us | 3 | no | no |
| `reagan` | Reagan Bombs Libya | us | 2 | no | no |
| `starwars` | Star Wars | us | 2 | no | no |
| `northseaoil` | North Sea Oil | us | 3 | no | no |
| `reformer` | The Reformer | ussr | 3 | no | no |
| `marine` | Marine Barracks Bombing | ussr | 2 | no | no |
| `KAL007` | Soviets Shoot Down KAL-007 | us | 4 | no | no |
| `glasnost` | Glasnost | ussr | 4 | no | no |
| `ortega` | Ortega Elected in Nicaragua | ussr | 2 | no | no |
| `terrorism` | Terrorism | both | 2 | no | yes |
| `irancontra` | Iran Contra Scandal | ussr | 2 | no | no |
| `chernobyl` | Chernobyl | us | 3 | no | no |
| `debtcrisis` | Latin American Debt Crisis | ussr | 2 | no | yes |
| `teardown` | Tear Down this Wall | us | 3 | no | no |
| `evilempire` | An Evil Empire | us | 3 | no | no |
| `aldrichames` | Aldrich Ames Remix | ussr | 3 | no | no |
| `pershing` | Pershing II Deployed | ussr | 3 | no | no |
| `wargames` | Wargames | both | 4 | no | no |
| `solidarity` | Solidarity | us | 2 | no | no |
| `iraniraq` | Iran-Iraq War | both | 2 | no | no |
| `yuri` | Yuri and Samantha | ussr | 2 | no | no |
| `awacs` | AWACS Sale to Saudis | us | 3 | no | no |

### Special

| ID | Name | Side | Ops | Scoring | Recurring |
| --- | --- | --- | ---: | --- | --- |
| `china` | China | both | 4 | no | yes |

## Notes

- Use the `ID` column in machine-readable warmup data.
- Display names are for audit only and may contain punctuation or capitalization that should not be parsed as IDs.
- `Initial US` and `Initial USSR` are the standard starting influence values before setup placement and before the US optional +2 bonus.
- `Battleground` and `Stability` are included to make hand-authored logs easier to validate.
