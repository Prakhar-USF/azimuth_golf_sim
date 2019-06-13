select * from public."user";
select * from public.shot;
select * from public.trajectory;
select * from public.files;
select * from public.contest;
select * from public.user_contest;

-- DROP TABLE public."user" cascade;
-- DROP TABLE public.shot cascade;
-- DROP TABLE public.trajectory cascade;
-- DROP TABLE public.files cascade;
-- DROP TABLE public.contest cascade;
-- DROP TABLE public.user_contest cascade;


-- delete from public."user" where id=8;
-- delete from public."user" where id=9;
-- delete from public."user" where id=10;
-- delete from public."user" where id=7;


-- UPDATE public."user"
-- SET subscription = 'premium'
-- Where id = 11;

-- INSERT INTO public.contest
-- VALUES (1, 'Sample Contest', 'no description', '2019-04-30', '2019-12-31', 120, 10);
-- INSERT INTO public.contest
-- VALUES (2, 'Another Sample Contest', 'no description', '2019-05-05', '2019-06-28', 50, -3);


-- INSERT INTO public.user_contest
-- VALUES (2, 1, 1, 1, '2019-04-30');
-- INSERT INTO public.user_contest
-- VALUES (1, 1, 2, 1, '2019-04-30');