DROP TABLE IF EXISTS mainstems;
CREATE TABLE mainstems (NHDPlusID REAL);

WITH RECURSIVE RecursiveJoin AS (
	SELECT
        ds.nhdplusid,
        ds.hydroseq,
        ds.dnhydroseq,
		ds.reachcode
    FROM
        NHDPlusFlowlineVAA us
	JOIN
		NHDPlusFlowlineVAA ds
	ON
		us.dnhydroseq=ds.hydroseq
	GROUP BY ds.nhdplusid
		HAVING MAX(us.totdasqkm) < 5.18 AND ds.totdasqkm >= 5.18

    UNION

    SELECT
        ds.nhdplusid,
        ds.hydroseq,
        ds.dnhydroseq,
		ds.reachcode
    FROM
        NHDPlusFlowlineVAA ds
    JOIN
        RecursiveJoin us
    ON
        us.dnhydroseq = ds.hydroseq
)

INSERT INTO mainstems (NHDPlusID)
SELECT nhdplusid 
FROM RecursiveJoin;

ALTER TABLE nhdplusflowlinevaa ADD COLUMN mainstem INTEGER;
UPDATE nhdplusflowlinevaa SET mainstem = 
CASE
	WHEN nhdplusid IN (SELECT nhdplusid FROM mainstems) THEN 1
	ELSE 0
END;
